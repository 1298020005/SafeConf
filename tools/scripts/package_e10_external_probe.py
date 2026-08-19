#!/usr/bin/env python3
"""Package the E10 external probe run into a compact docs artifact.

The raw probe directory contains matrices and run-time internals that are useful
for debugging but noisy for Git/history.  This packager keeps the human-facing
pieces: key tables, key figures, an honest summary, and the exact command/path
needed to reproduce the run.
"""

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
RUNTIME = ROOT / "runtime" / "e10_external_probe_kcc_haber_parekh_20260707"
OUT = ROOT / "docs" / "实验结果" / "E10_external_task_validation_probe_20260707"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"

TABLE_FILES = [
    "DATASET_TASK_SUMMARY.csv",
    "MAIN_PER_DATASET_SUMMARY.csv",
    "CONFIDENCE_EVAL_SUMMARY.csv",
    "RISK_COVERAGE.csv",
    "HELDOUT_PAIR_SPLIT_SUMMARY.csv",
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

REPORT_FILES = [
    "MVP_V2_1_REPORT.md",
    "stage_completion_checklist_v2_1.md",
]


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(RUNTIME / "tables" / name)


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    ).stdout.strip()


def safe_copy(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def pct(x: Any, digits: int = 2) -> str:
    if pd.isna(x):
        return ""
    return f"{float(x):.{digits}f}%"


def num(x: Any, digits: int = 3) -> str:
    if pd.isna(x):
        return ""
    return f"{float(x):.{digits}f}"


def md_table(df: pd.DataFrame, cols: list[str], rename: dict[str, str] | None = None) -> str:
    rename = rename or {}
    show = df[cols].rename(columns=rename).copy()
    headers = [str(c) for c in show.columns]

    def cell(v: Any) -> str:
        if pd.isna(v):
            return ""
        text = str(v).replace("\n", " ").replace("|", "\\|")
        return text

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(cell(row[c]) for c in show.columns) + " |")
    return "\n".join(lines)


def df_html(df: pd.DataFrame, cols: list[str], n: int = 20) -> str:
    return df[cols].head(n).to_html(index=False, escape=True)


def parse_stage_checklist() -> tuple[int, int, str]:
    path = RUNTIME / "stage_completion_checklist_v2_1.md"
    text = path.read_text(encoding="utf-8")
    passed = text.count("| PASS |")
    failed = text.count("| FAIL |")
    return passed, failed, text


def build_summaries() -> dict[str, Any]:
    dataset = read_csv("DATASET_TASK_SUMMARY.csv")
    main = read_csv("MAIN_PER_DATASET_SUMMARY.csv")
    eval_df = read_csv("CONFIDENCE_EVAL_SUMMARY.csv")
    split = read_csv("HELDOUT_PAIR_SPLIT_SUMMARY.csv")
    missing = read_csv("CONFIDENCE_FEATURE_MISSINGNESS_BY_DATASET.csv")
    transfer = read_csv("TRANSFERABILITY_RANKING.csv")

    overall = eval_df[eval_df["level"].eq("overall")].copy()
    if overall.empty:
        overall = eval_df.copy()
    best_overall = overall.sort_values(
        ["direction_aligned_spearman", "risk_cov_80_improve_pct"],
        ascending=[False, False],
    ).iloc[0].to_dict()

    dataset_best = (
        eval_df[eval_df["level"].eq("dataset")]
        .sort_values(["dataset_name", "direction_aligned_spearman"], ascending=[True, False])
        .groupby("dataset_name", as_index=False)
        .head(1)
        .sort_values("dataset_name")
    )

    split_summary = {
        "folds": int(split["fold_id"].nunique()),
        "support_check_pass": bool(split["support_check_pass"].fillna(False).all()),
        "total_test_tasks": int(split["n_test"].sum()),
        "total_train_tasks": int(split["n_train"].sum()),
        "total_val_tasks": int(split["n_val"].sum()),
    }
    passed, failed, checklist_text = parse_stage_checklist()

    return {
        "dataset": dataset,
        "main": main,
        "eval": eval_df,
        "missing": missing,
        "transfer": transfer,
        "best_overall": best_overall,
        "dataset_best": dataset_best,
        "split_summary": split_summary,
        "checklist_passed": passed,
        "checklist_failed": failed,
        "checklist_text": checklist_text,
    }


def write_reports(summary: dict[str, Any]) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    best = summary["best_overall"]
    dataset = summary["dataset"]
    main = summary["main"]
    dataset_best = summary["dataset_best"]
    missing = summary["missing"]
    split = summary["split_summary"]

    report = f"""# E10 外部任务级验证探针报告

生成时间：{now}

## 1. 一句话结论

E10 已经从外部 context-generalization 数据中跑通一个小型任务级探针：`KaggleCrossCell`、`Haber`、`Parekh` 共 {int(dataset['n_tasks'].sum())} 个任务、{split['total_test_tasks']} 个测试任务，split 支持检查通过。最强信号来自 `model_disagreement_risk`，overall aligned Spearman = {num(best['direction_aligned_spearman'])}，80% coverage 下平均 RMSE 改善 {pct(best['risk_cov_80_improve_pct'])}。

这不是最终可投稿的外部验证结论。它更像“侦察兵”：证明管线能跑，也暴露出 frozen/simple combined 在 KaggleCrossCell 上较弱、learned risk 未超过 disagreement 的问题。

## 2. 数据与任务

{md_table(dataset, ['dataset','n_tasks','n_contexts','n_perturbations','context_col','perturbation_col','n_genes'])}

## 3. 关键结果

### 3.1 每个数据集的主要分数

{md_table(main, [
    'dataset_name',
    'simple_combined_confidence_aligned_rho',
    'simple_combined_confidence_risk_cov_80_improve_pct',
    'learned_risk_score_aligned_rho',
    'learned_risk_score_risk_cov_80_improve_pct',
    'model_disagreement_risk_aligned_rho',
    'model_disagreement_risk_risk_cov_80_improve_pct',
])}

### 3.2 每个数据集当前最强单项信号

{md_table(dataset_best, [
    'dataset_name',
    'score_name',
    'score_type',
    'n',
    'direction_aligned_spearman',
    'risk_cov_80_improve_pct',
    'high_low_rmse_gap',
])}

## 4. 失败边界

- `KaggleCrossCell` 上 simple combined aligned rho = 0.121，80% coverage 改善为 -0.55%，不够稳。
- learned risk overall 未超过 model disagreement，说明现有轻量学习器暂时不是 E10 的主角。
- `perturbation_effect_stability` 在 KaggleCrossCell 缺失率 {pct(missing.loc[missing['dataset_name'].eq('KaggleCrossCell'), 'perturbation_effect_stability'].iloc[0] * 100)}，在 Parekh 缺失率 {pct(missing.loc[missing['dataset_name'].eq('Parekh'), 'perturbation_effect_stability'].iloc[0] * 100)}，解释了组合分数在部分外部数据上不够稳定。

## 5. 图件

- `figures/F1_task_schematic.png`：任务定义
- `figures/F2_context_perturbation_matrix.png`：context × perturbation 覆盖
- `figures/F3_prediction_record_flow.png`：PredictionRecord 流程
- `figures/F4_confidence_vs_true_error_scatter.png`：分数与真实误差散点
- `figures/F5_risk_coverage_curve.png`：risk-coverage 曲线
- `figures/F7_per_dataset_spearman_comparison.png`：每数据集 Spearman 对比
- `figures/F10_transferability_ranking.png`：外部迁移排序

## 6. 下一步判断

1. E10 下一轮不急着扩成大模型训练，先固定一个清晰外部 split，比较 task-only、disagreement、magnitude、task+model combined。
2. 若目标是一区/CCF-A，E10 需要变成“跨数据源的 selective prediction / risk routing”证据，而不是只证明某个组合分数 Spearman 为正。
3. 对 KaggleCrossCell 需要单独查失败原因：context 相似性方向反了，stability 大量缺失，说明外部数据字段和主表协议不完全同构。

## 7. 复现

本次运行原始目录：

```text
runtime/e10_external_probe_kcc_haber_parekh_20260707
```

原始脚本：

```text
code/20260426_154505_perturb_transport_final_push/confidence_task/run_confidence_mvp_v2_1.py
```
"""
    (REPORTS / "E10_EXTERNAL_PROBE_REPORT.md").write_text(report, encoding="utf-8")

    cards = [
        ("外部任务数", str(int(dataset["n_tasks"].sum()))),
        ("测试任务数", str(split["total_test_tasks"])),
        ("最强 overall rho", num(best["direction_aligned_spearman"])),
        ("80% coverage 改善", pct(best["risk_cov_80_improve_pct"])),
        ("Gate", f"{summary['checklist_passed']} PASS / {summary['checklist_failed']} FAIL"),
    ]
    card_html = "".join(
        f'<div class="metric"><div class="k">{html.escape(k)}</div><div class="v">{html.escape(v)}</div></div>'
        for k, v in cards
    )
    fig_html = "\n".join(
        f'<figure><img src="../figures/{html.escape(name)}" alt="{html.escape(name)}"><figcaption>{html.escape(name)}</figcaption></figure>'
        for name in FIGURE_FILES
    )

    html_doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>E10 外部任务级验证探针</title>
<style>
:root {{
  --ink:#17201c;
  --muted:#66736d;
  --line:#dfe7e2;
  --accent:#2f6f5e;
  --soft:#f6f8f4;
  --warn:#a65d00;
}}
body {{
  margin:0;
  background:#fbfbf8;
  color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC","Microsoft YaHei",Arial,sans-serif;
  line-height:1.72;
}}
.wrap {{ max-width:1180px; margin:0 auto; padding:34px 30px 70px; }}
.hero {{ border-bottom:1px solid var(--line); padding:30px 0 24px; }}
h1 {{ font-size:34px; margin:0 0 10px; letter-spacing:.02em; }}
h2 {{ margin-top:34px; padding-bottom:8px; border-bottom:2px solid var(--line); }}
p {{ margin:9px 0; }}
.lead {{ color:var(--muted); font-size:17px; max-width:900px; }}
.metrics {{ display:grid; grid-template-columns:repeat(5,1fr); gap:12px; margin:24px 0; }}
.metric {{ background:white; border:1px solid var(--line); border-radius:14px; padding:16px; }}
.metric .k {{ color:var(--muted); font-size:13px; }}
.metric .v {{ color:var(--accent); font-size:26px; font-weight:720; margin-top:4px; }}
.card {{ background:white; border:1px solid var(--line); border-radius:16px; padding:20px; margin:18px 0; overflow-x:auto; }}
.honest {{ border-left:5px solid var(--warn); }}
table {{ border-collapse:collapse; width:100%; font-size:13px; }}
th,td {{ border:1px solid var(--line); padding:7px 8px; vertical-align:top; }}
th {{ background:#f1f5f2; text-align:left; }}
.figgrid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }}
figure {{ margin:0; background:white; border:1px solid var(--line); border-radius:16px; padding:12px; }}
img {{ max-width:100%; display:block; border-radius:10px; background:white; }}
figcaption {{ color:var(--muted); font-size:13px; margin-top:8px; }}
code {{ background:#eef3ef; padding:2px 6px; border-radius:5px; }}
@media (max-width:900px) {{ .metrics,.figgrid {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<main class="wrap">
<section class="hero">
  <h1>E10 外部任务级验证探针</h1>
  <p class="lead">本次先确认外部数据能否进入 SafeConf 的任务级风险审计闭环。当前结论：管线跑通，disagreement 信号强，组合分数在部分外部数据上需要重查。</p>
  <div class="metrics">{card_html}</div>
</section>

<section class="card">
  <h2>1. 数据与任务</h2>
  {df_html(dataset, ['dataset','n_tasks','n_contexts','n_perturbations','context_col','perturbation_col','n_genes'])}
</section>

<section class="card">
  <h2>2. 每数据集主结果</h2>
  {df_html(main, [
    'dataset_name',
    'simple_combined_confidence_aligned_rho',
    'simple_combined_confidence_risk_cov_80_improve_pct',
    'learned_risk_score_aligned_rho',
    'learned_risk_score_risk_cov_80_improve_pct',
    'model_disagreement_risk_aligned_rho',
    'model_disagreement_risk_risk_cov_80_improve_pct',
  ])}
</section>

<section class="card">
  <h2>3. 每数据集最强单项信号</h2>
  {df_html(dataset_best, ['dataset_name','score_name','score_type','n','direction_aligned_spearman','risk_cov_80_improve_pct','high_low_rmse_gap'])}
</section>

<section class="card honest">
  <h2>4. 失败边界</h2>
  <p>KaggleCrossCell 上 simple combined 只有 0.121；learned risk 没有超过 model disagreement；外部数据中 perturbation stability 有明显缺失。这三点不能藏，要在下一轮实验里正面处理。</p>
</section>

<section>
  <h2>5. 图件</h2>
  <div class="figgrid">{fig_html}</div>
</section>
</main>
</body>
</html>
"""
    (REPORTS / "E10_EXTERNAL_PROBE.html").write_text(html_doc, encoding="utf-8")


def main() -> None:
    if not RUNTIME.exists():
        raise SystemExit(f"Missing runtime directory: {RUNTIME}")
    for d in (TABLES, FIGURES, REPORTS):
        d.mkdir(parents=True, exist_ok=True)

    for name in TABLE_FILES:
        safe_copy(RUNTIME / "tables" / name, TABLES / name)
    for name in FIGURE_FILES:
        safe_copy(RUNTIME / "figures" / name, FIGURES / name)
    for name in REPORT_FILES:
        safe_copy(RUNTIME / name, REPORTS / name)

    summary = build_summaries()
    write_reports(summary)

    status = {
        "status": "ok",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "output_dir": str(OUT.relative_to(ROOT)),
        "runtime_source": str(RUNTIME.relative_to(ROOT)),
        "input_git_commit": git_head(),
        "copied_tables": TABLE_FILES,
        "copied_figures": FIGURE_FILES,
        "interpretation": {
            "best_overall_score": summary["best_overall"].get("score_name"),
            "best_overall_aligned_spearman": float(summary["best_overall"].get("direction_aligned_spearman")),
            "best_overall_risk_cov_80_improve_pct": float(summary["best_overall"].get("risk_cov_80_improve_pct")),
            "stage_checklist_passed": int(summary["checklist_passed"]),
            "stage_checklist_failed": int(summary["checklist_failed"]),
        },
    }
    (OUT / "RUN_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    readme = """# E10 external task-level validation probe

先看：

- `reports/E10_EXTERNAL_PROBE.html`
- `reports/E10_EXTERNAL_PROBE_REPORT.md`

说明：

这个目录只保留可汇报、可审计的轻量文件。原始中间矩阵仍在 `runtime/e10_external_probe_kcc_haber_parekh_20260707/`。

重新打包：

```bash
python3 tools/scripts/package_e10_external_probe.py
```
"""
    (OUT / "README_先看这个.md").write_text(readme, encoding="utf-8")
    print(f"Wrote E10 probe package to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
