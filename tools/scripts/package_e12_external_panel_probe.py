#!/usr/bin/env python3
"""Package the E12 expanded external-panel probe into docs."""

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
RUNTIME = ROOT / "runtime" / "e12_external_panel_probe_20260707"
OUT = ROOT / "docs" / "实验结果" / "E12_external_panel_probe_20260707"
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


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    ).stdout.strip()


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def cell(v: Any, digits: int = 6) -> str:
    if pd.isna(v):
        return ""
    if isinstance(v, float):
        return f"{v:.{digits}g}"
    return str(v).replace("\n", " ").replace("|", "/")


def md_table(df: pd.DataFrame, cols: list[str], n: int = 80) -> str:
    show = df[cols].head(n).copy()
    lines = [
        "| " + " | ".join(map(str, show.columns)) + " |",
        "| " + " | ".join(["---"] * len(show.columns)) + " |",
    ]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(cell(row[c]) for c in show.columns) + " |")
    if len(df) > n:
        lines.append(f"\n_Only first {n} of {len(df)} rows shown._")
    return "\n".join(lines)


def html_table(df: pd.DataFrame, cols: list[str], n: int = 30) -> str:
    return df[cols].head(n).to_html(index=False, escape=True)


def build_summary() -> dict[str, Any]:
    dataset = read_csv("DATASET_TASK_SUMMARY.csv")
    split = read_csv("HELDOUT_PAIR_SPLIT_SUMMARY.csv")
    main = read_csv("MAIN_PER_DATASET_SUMMARY.csv")
    eval_df = read_csv("CONFIDENCE_EVAL_SUMMARY.csv")
    missing = read_csv("CONFIDENCE_FEATURE_MISSINGNESS_BY_DATASET.csv")

    overall = eval_df[eval_df["level"].eq("overall")].copy()
    overall = overall.sort_values(
        ["direction_aligned_spearman", "risk_cov_80_improve_pct"],
        ascending=[False, False],
    )
    best = overall.iloc[0].to_dict()
    simple = overall[overall["score_name"].eq("simple_combined_confidence")].iloc[0].to_dict()
    learned = overall[overall["score_name"].eq("learned_risk_score")].iloc[0].to_dict()

    test_by_dataset = split.groupby("dataset_name", as_index=False)["n_test"].sum()
    evaluable = test_by_dataset[test_by_dataset["n_test"] > 0].copy()
    skipped = test_by_dataset[test_by_dataset["n_test"] <= 0].copy()
    task_join = dataset.merge(test_by_dataset, left_on="dataset", right_on="dataset_name", how="left")
    task_join["e12_status"] = task_join["n_test"].fillna(0).map(
        lambda x: "evaluable" if int(x) > 0 else "not_evaluable_under_heldout_pair"
    )

    checklist_text = (RUNTIME / "stage_completion_checklist_v2_1.md").read_text(encoding="utf-8")
    passed = checklist_text.count("| PASS |")
    failed = checklist_text.count("| FAIL |")
    skipped_gate = checklist_text.count("| SKIPPED")

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
        "task_join": task_join,
        "evaluable": evaluable,
        "skipped": skipped,
        "checklist_text": checklist_text,
        "gate_passed": passed,
        "gate_failed": failed,
        "gate_skipped": skipped_gate,
    }


def write_reports(s: dict[str, Any]) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    best = s["best"]
    simple = s["simple"]
    learned = s["learned"]
    task_join = s["task_join"]
    main = s["main"]
    overall = s["overall"]
    missing = s["missing"]

    report = f"""# E12 外部面板扩展探针报告

生成时间：{now}

## 1. 结论

E12 在 6 个外部候选数据集上启动任务级探针。`KaggleCrossPatient`、`crossPatient`、`sciplex3` 可进入 held-out pair 测试；`kangCrossCell`、`kangCrossPatient`、`TCDD` 因只有 1 个 perturbation，在当前 held-out pair 规则下没有 test pair。

可评估部分共 308 条 test 记录。overall 最强信号是 `{best['score_name']}`：aligned Spearman = {float(best['direction_aligned_spearman']):.3f}，80% coverage 平均 RMSE 改善 = {float(best['risk_cov_80_improve_pct']):.2f}%。`simple_combined_confidence` 的 aligned Spearman = {float(simple['direction_aligned_spearman']):.3f}，80% coverage 改善 = {float(simple['risk_cov_80_improve_pct']):.2f}%。`learned_risk_score` 为正但弱于 disagreement：aligned Spearman = {float(learned['direction_aligned_spearman']):.3f}。

## 2. 数据集适配结果

{md_table(task_join, ['dataset','n_tasks','n_contexts','n_perturbations','context_col','perturbation_col','n_test','e12_status'])}

## 3. 每数据集主结果

{md_table(main, [
    'dataset_name',
    'simple_combined_confidence_aligned_rho',
    'simple_combined_confidence_risk_cov_80_improve_pct',
    'learned_risk_score_aligned_rho',
    'learned_risk_score_risk_cov_80_improve_pct',
    'model_disagreement_risk_aligned_rho',
    'model_disagreement_risk_risk_cov_80_improve_pct',
])}

## 4. Overall 信号排序

{md_table(overall, ['score_name','score_type','n','direction_aligned_spearman','risk_cov_80_improve_pct','high_low_rmse_gap'])}

## 5. 必须保留的边界

- E12 实际可评估数据集为 3 个；另外 3 个候选需要换验证定义。
- `sciplex3` 是当前最强外部 chemical 信号：model disagreement aligned rho = 0.885。
- `crossPatient` 的 simple combined 只有 0.075，说明组合分数跨数据集仍不稳。
- learned risk 没有超过 model disagreement；这与 E10 一致。
- 当前仍是轻量参考预测器探针；GEARS/CPA/scGPT 逐模型向量验证尚未完成。

## 6. 下一步

1. 对 `sciplex3` 单独做 chemical-focused frozen split，保留 disagreement、support、simple combined、magnitude 的统一比较。
2. 对 `KaggleCrossPatient` 做 context/patient 层解释，检查 support_count 与 context similarity 为什么比 learned 更稳定。
3. 对 `kangCrossCell/kangCrossPatient/TCDD` 改用 leave-context-out 或 chemical dose split；held-out pair 不适合只有一个 perturbation 的数据。
"""
    (REPORTS / "E12_EXTERNAL_PANEL_PROBE_REPORT.md").write_text(report, encoding="utf-8")

    cards = [
        ("候选数据集", str(len(task_join))),
        ("可评估数据集", str(int((task_join["e12_status"] == "evaluable").sum()))),
        ("test 记录", str(int(overall["n"].max()))),
        ("best rho", f"{float(best['direction_aligned_spearman']):.3f}"),
        ("simple cov80", f"{float(simple['risk_cov_80_improve_pct']):.2f}%"),
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
<head>
<meta charset="utf-8">
<title>E12 外部面板扩展探针</title>
<style>
:root{{--ink:#17201c;--muted:#66736d;--line:#dfe7e2;--accent:#2f6f5e;--soft:#f8faf7;--warn:#9a5b00}}
body{{margin:0;background:#fbfbf8;color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC","Microsoft YaHei",Arial,sans-serif;line-height:1.72}}
.wrap{{max-width:1180px;margin:0 auto;padding:34px 30px 70px}}
.hero{{border-bottom:1px solid var(--line);padding:28px 0 22px}}
h1{{font-size:34px;margin:0 0 10px}}h2{{margin-top:32px;padding-bottom:8px;border-bottom:2px solid var(--line)}}
.lead{{max-width:930px;color:var(--muted);font-size:17px}}
.metrics{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:24px 0}}
.metric,.card,figure{{background:white;border:1px solid var(--line);border-radius:15px}}
.metric{{padding:16px}}.metric .k{{color:var(--muted);font-size:13px}}.metric .v{{color:var(--accent);font-size:26px;font-weight:720}}
.card{{padding:20px;margin:18px 0;overflow-x:auto}}.warn{{border-left:5px solid var(--warn)}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid var(--line);padding:7px 8px;vertical-align:top}}th{{background:#f1f5f2;text-align:left}}
.figgrid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}}figure{{margin:0;padding:12px}}img{{max-width:100%;display:block;background:white;border-radius:10px}}figcaption{{color:var(--muted);font-size:13px;margin-top:8px}}
@media(max-width:900px){{.metrics,.figgrid{{grid-template-columns:1fr}}}}
</style>
</head>
<body><main class="wrap">
<section class="hero"><h1>E12 外部面板扩展探针</h1><p class="lead">这一步把 E10 从 3 个外部数据集扩展到 6 个候选。重点是筛清楚哪些数据适合 held-out pair，哪些需要换验证定义。</p><div class="metrics">{cards_html}</div></section>
<section class="card"><h2>1. 数据集适配</h2>{html_table(task_join, ['dataset','n_tasks','n_contexts','n_perturbations','context_col','perturbation_col','n_test','e12_status'])}</section>
<section class="card"><h2>2. 每数据集主结果</h2>{html_table(main, ['dataset_name','simple_combined_confidence_aligned_rho','simple_combined_confidence_risk_cov_80_improve_pct','learned_risk_score_aligned_rho','learned_risk_score_risk_cov_80_improve_pct','model_disagreement_risk_aligned_rho','model_disagreement_risk_risk_cov_80_improve_pct'])}</section>
<section class="card"><h2>3. Overall 信号排序</h2>{html_table(overall, ['score_name','score_type','n','direction_aligned_spearman','risk_cov_80_improve_pct','high_low_rmse_gap'])}</section>
<section class="card warn"><h2>4. 解释边界</h2><p>E12 有 6 个候选，但只有 3 个可按 held-out pair 评价。sciplex3 信号很强，crossPatient simple combined 很弱；learned risk 仍没有超过 disagreement。</p></section>
<section><h2>5. 图件</h2><div class="figgrid">{fig_html}</div></section>
</main></body></html>
"""
    (REPORTS / "E12_EXTERNAL_PANEL_PROBE.html").write_text(html_doc, encoding="utf-8")

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
        "evaluable_datasets": summary["evaluable"]["dataset_name"].tolist(),
        "not_evaluable_under_heldout_pair": summary["skipped"]["dataset_name"].tolist(),
    }
    (OUT / "RUN_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "README_先看这个.md").write_text(
        "# E12 external panel probe\n\n"
        "先看：\n\n"
        "- `reports/E12_EXTERNAL_PANEL_PROBE.html`\n"
        "- `reports/E12_EXTERNAL_PANEL_PROBE_REPORT.md`\n\n"
        "这个目录只保存可汇报的轻量结果。原始矩阵在 `runtime/e12_external_panel_probe_20260707/`。\n",
        encoding="utf-8",
    )
    print(f"Wrote E12 package to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
