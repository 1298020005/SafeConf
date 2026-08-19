#!/usr/bin/env python3
"""Package E13 official sciplex3 3-cell-line focused panel."""

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
RUNTIME = ROOT / "runtime" / "e13_sciplex3_official_3cell_panel_focused_20260707"
OUT = ROOT / "docs" / "实验结果" / "E13_sciplex3_official_3cell_panel_20260707"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"

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


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(RUNTIME / "tables" / name)


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    ).stdout.strip()


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


def build_summary() -> dict[str, Any]:
    dataset = read_csv("DATASET_TASK_SUMMARY.csv")
    split = read_csv("HELDOUT_PAIR_SPLIT_SUMMARY.csv")
    main = read_csv("MAIN_PER_DATASET_SUMMARY.csv")
    eval_df = read_csv("CONFIDENCE_EVAL_SUMMARY.csv")
    missing = read_csv("CONFIDENCE_FEATURE_MISSINGNESS_BY_DATASET.csv")
    overall = eval_df[eval_df["level"].eq("overall")].sort_values(
        ["direction_aligned_spearman", "risk_cov_80_improve_pct"],
        ascending=[False, False],
    )
    best = overall.iloc[0].to_dict()
    simple = overall[overall["score_name"].eq("simple_combined_confidence")].iloc[0].to_dict()
    learned = overall[overall["score_name"].eq("learned_risk_score")].iloc[0].to_dict()
    disagreement = overall[overall["score_name"].eq("model_disagreement_risk")].iloc[0].to_dict()
    magnitude = overall[overall["score_name"].eq("prediction_magnitude_risk")].iloc[0].to_dict()
    checklist = (RUNTIME / "stage_completion_checklist_v2_1.md").read_text(encoding="utf-8")
    return {
        "dataset": dataset,
        "split": split,
        "main": main,
        "eval": eval_df,
        "missing": missing,
        "overall": overall,
        "best": best,
        "simple": simple,
        "learned": learned,
        "disagreement": disagreement,
        "magnitude": magnitude,
        "gate_passed": checklist.count("| PASS |"),
        "gate_failed": checklist.count("| FAIL |"),
        "gate_skipped": checklist.count("| SKIPPED"),
    }


def write_reports(s: dict[str, Any]) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    dataset = s["dataset"]
    split = s["split"]
    main = s["main"]
    overall = s["overall"]
    best = s["best"]
    simple = s["simple"]
    learned = s["learned"]
    disagreement = s["disagreement"]
    magnitude = s["magnitude"]

    report = f"""# E13 官方 sciplex3 三细胞系 focused panel

生成时间：{now}

## 1. 结论

E13 把官方 `sciplex3_A549.h5ad`、`sciplex3_K562.h5ad`、`sciplex3_MCF7.h5ad` 合成为一个三细胞系任务面板：context 为 cell line，perturbation 为 `drug_dose_name`。原始可构建 2,237 个任务，其中 743 个 drug-dose 同时出现在三种细胞系。为避免高维最近邻计算拖垮流程，本轮按每个 drug-dose 的最小细胞数与总细胞数选取 top 80，形成 240 个任务。

正式 test 记录 480 条，无 pair/context/perturbation leakage。最强信号为 `{best['score_name']}`：aligned Spearman = {float(best['direction_aligned_spearman']):.3f}，80% coverage RMSE 改善 = {float(best['risk_cov_80_improve_pct']):.2f}%。`learned_risk_score` 与 disagreement 非常接近：aligned Spearman = {float(learned['direction_aligned_spearman']):.3f}；`simple_combined_confidence` aligned Spearman = {float(simple['direction_aligned_spearman']):.3f}。

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

## 6. 必须保留的边界

- 本轮为 focused top-80 shared drug-dose panel；743 shared drug-dose 全量版本留待优化后复跑。
- 全量版本第一次尝试在高维最近邻 feature 阶段过慢，已中断；后续需要优化 `compute_features` 后再跑。
- `prediction_magnitude_risk` 在该 panel 上方向为负：aligned Spearman = {float(magnitude['direction_aligned_spearman']):.3f}。
- `perturbation_effect_stability` 缺失率为 55.6%，组合分数受该特征覆盖影响。
- 当前仍使用两个轻量参考预测器；GEARS/CPA/scGPT 多模型逐向量验证尚未完成。

## 7. 下一步

1. 优化 `compute_features` 中 context cosine / OOD nearest distance 的矩阵化实现。
2. 复跑官方 sciplex3 全 743 shared drug-dose panel。
3. 把 GEARS/CPA/scGPT 的同任务预测向量接入该三细胞系 panel，比较 task-only、model-only、task+model combined。
"""
    (REPORTS / "E13_SCIPLEX3_OFFICIAL_3CELL_REPORT.md").write_text(report, encoding="utf-8")

    cards = [
        ("任务", str(int(dataset["n_tasks"].iloc[0]))),
        ("contexts", str(int(dataset["n_contexts"].iloc[0]))),
        ("selected drug-dose", str(int(dataset["n_selected_perturbations"].iloc[0]))),
        ("best rho", f"{float(best['direction_aligned_spearman']):.3f}"),
        ("learned rho", f"{float(learned['direction_aligned_spearman']):.3f}"),
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
<head><meta charset="utf-8"><title>E13 官方 sciplex3 三细胞系 focused panel</title>
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
<section class="hero"><h1>E13 官方 sciplex3 三细胞系 focused panel</h1><p class="lead">把 A549、K562、MCF7 三个官方 sciplex3 文件合成一个 cell-line context 面板，验证官方 chemical drug-dose 外部任务风险信号。</p><div class="metrics">{cards_html}</div></section>
<section class="card"><h2>1. 面板构建</h2>{html_table(dataset, ['dataset','n_tasks','n_contexts','n_perturbations','n_raw_tasks','n_shared_perturbations','n_selected_perturbations','selection_rule'])}</section>
<section class="card"><h2>2. 主结果</h2>{html_table(main, ['dataset_name','simple_combined_confidence_aligned_rho','simple_combined_confidence_risk_cov_80_improve_pct','learned_risk_score_aligned_rho','learned_risk_score_risk_cov_80_improve_pct','model_disagreement_risk_aligned_rho','model_disagreement_risk_risk_cov_80_improve_pct'])}</section>
<section class="card"><h2>3. Overall 信号排序</h2>{html_table(overall, ['score_name','score_type','n','direction_aligned_spearman','risk_cov_80_improve_pct','high_low_rmse_gap'])}</section>
<section class="card warn"><h2>4. 边界</h2><p>本轮是 top-80 shared drug-dose focused panel。全量 743 shared drug-dose 需要先优化高维 feature 计算。</p></section>
<section><h2>5. 图件</h2><div class="figgrid">{fig_html}</div></section>
</main></body></html>
"""
    (REPORTS / "E13_SCIPLEX3_OFFICIAL_3CELL.html").write_text(html_doc, encoding="utf-8")
    copy_file(RUNTIME / "MVP_V2_1_REPORT.md", REPORTS / "MVP_V2_1_REPORT.md")
    copy_file(RUNTIME / "stage_completion_checklist_v2_1.md", REPORTS / "stage_completion_checklist_v2_1.md")


def main() -> None:
    if not RUNTIME.exists():
        raise SystemExit(f"Missing runtime directory: {RUNTIME}")
    for p in (TABLES, FIGURES, REPORTS):
        p.mkdir(parents=True, exist_ok=True)
    for name in TABLE_FILES:
        copy_file(RUNTIME / "tables" / name, TABLES / name)
    for name in FIGURE_FILES:
        copy_file(RUNTIME / "figures" / name, FIGURES / name)
    summary = build_summary()
    write_reports(summary)
    status = {
        "status": "ok",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "output_dir": str(OUT.relative_to(ROOT)),
        "runtime_source": str(RUNTIME.relative_to(ROOT)),
        "input_git_commit": git_head(),
        "best_overall_score": summary["best"]["score_name"],
        "best_overall_aligned_spearman": float(summary["best"]["direction_aligned_spearman"]),
        "best_overall_risk_cov_80_improve_pct": float(summary["best"]["risk_cov_80_improve_pct"]),
        "learned_aligned_spearman": float(summary["learned"]["direction_aligned_spearman"]),
        "simple_combined_aligned_spearman": float(summary["simple"]["direction_aligned_spearman"]),
        "panel": {
            "n_tasks": int(summary["dataset"]["n_tasks"].iloc[0]),
            "n_contexts": int(summary["dataset"]["n_contexts"].iloc[0]),
            "n_shared_perturbations": int(summary["dataset"]["n_shared_perturbations"].iloc[0]),
            "n_selected_perturbations": int(summary["dataset"]["n_selected_perturbations"].iloc[0]),
        },
    }
    (OUT / "RUN_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "README_先看这个.md").write_text(
        "# E13 sciplex3 official 3-cell-line panel\n\n"
        "先看：\n\n"
        "- `reports/E13_SCIPLEX3_OFFICIAL_3CELL.html`\n"
        "- `reports/E13_SCIPLEX3_OFFICIAL_3CELL_REPORT.md`\n\n"
        "这个目录只保存可汇报的轻量结果。原始矩阵在 `runtime/e13_sciplex3_official_3cell_panel_focused_20260707/`。\n",
        encoding="utf-8",
    )
    print(f"Wrote E13 package to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
