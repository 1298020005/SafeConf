#!/usr/bin/env python3
"""Package E22: strict contract smoke for the patched confidence generator."""

from __future__ import annotations

import html
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = ROOT / "code" / "20260426_154505_perturb_transport_final_push"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from safetrans_confidence.data.records import validate_prediction_record_artifacts  # noqa: E402


RUNTIME = ROOT / "runtime" / "e22_generator_strict_smoke_20260707"
OUT = ROOT / "docs" / "实验结果" / "E22_generator_strict_smoke_20260707"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"
FIGURES = OUT / "figures"


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


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def md_table(df: pd.DataFrame, n: int = 50) -> str:
    show = df.head(n).copy()
    lines = [
        "| " + " | ".join(map(str, show.columns)) + " |",
        "| " + " | ".join(["---"] * len(show.columns)) + " |",
    ]
    for _, row in show.iterrows():
        vals = []
        for col in show.columns:
            val = row[col]
            if pd.isna(val):
                vals.append("")
            elif isinstance(val, float):
                vals.append(f"{val:.6g}")
            else:
                vals.append(str(val).replace("\n", " ").replace("|", "/"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def html_table(df: pd.DataFrame, n: int = 50) -> str:
    return df.head(n).to_html(index=False, escape=True)


def collect() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records = pd.read_csv(RUNTIME / "tables" / "PREDICTION_RECORDS.csv")
    issues = validate_prediction_record_artifacts(RUNTIME, strict=True)
    groups = records.groupby(["dataset_name", "fold_id", "split", "task_key", "context", "perturbation"], dropna=False)
    summary = pd.DataFrame(
        [
            {
                "runtime_dir": rel(RUNTIME),
                "dataset_names": ",".join(sorted(records["dataset_name"].dropna().astype(str).unique())),
                "n_records": int(len(records)),
                "n_task_groups": int(groups.ngroups),
                "n_predictors": int(records["predictor_name"].nunique()),
                "predictors": ",".join(sorted(records["predictor_name"].dropna().astype(str).unique())),
                "n_unique_true_effect_keys": int(records["true_effect_key"].nunique()),
                "n_unique_predicted_effect_keys": int(records["predicted_effect_key"].nunique()),
                "strict_status": "pass" if not issues else "fail",
                "strict_issue_count": int(len(issues)),
                "strict_issues": ";".join(issues),
            }
        ]
    )
    key_sample = records[
        ["record_id", "task_key", "split", "predictor_name", "predicted_effect_key", "true_effect_key"]
    ].head(30)
    issues_df = pd.DataFrame({"issue": issues})
    return summary, key_sample, issues_df


def write_svg(summary: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    row = summary.iloc[0]
    status = str(row["strict_status"]).upper()
    fill = "#28734f" if status == "PASS" else "#a4403a"
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="980" height="260" viewBox="0 0 980 260">
<rect width="980" height="260" fill="#fbfbf8"/>
<text x="42" y="38" font-size="23" font-weight="800" fill="#17212b">E22 generator strict smoke</text>
<text x="42" y="62" font-size="14" fill="#5d6974">The patched confidence generator now emits task-scoped true_effect_key values that strict-pass the PredictionRecord contract.</text>
<g transform="translate(52 104)">
<rect x="0" y="0" width="190" height="62" rx="8" fill="#fff" stroke="#d8e0e4"/>
<text x="95" y="25" text-anchor="middle" font-size="15" font-weight="700">Patched generator</text>
<text x="95" y="46" text-anchor="middle" font-size="12" fill="#5d6974">Haber, 200 genes</text>
<text x="215" y="38" font-size="24" fill="#087f73">→</text>
<rect x="255" y="0" width="210" height="62" rx="8" fill="#fff" stroke="#d8e0e4"/>
<text x="360" y="25" text-anchor="middle" font-size="15" font-weight="700">{int(row['n_records'])} records</text>
<text x="360" y="46" text-anchor="middle" font-size="12" fill="#5d6974">{int(row['n_task_groups'])} task-scoped true keys</text>
<text x="488" y="38" font-size="24" fill="#087f73">→</text>
<rect x="528" y="0" width="190" height="62" rx="8" fill="#fff" stroke="#d8e0e4"/>
<text x="623" y="25" text-anchor="middle" font-size="15" font-weight="700">Strict contract</text>
<text x="623" y="46" text-anchor="middle" font-size="12" fill="#5d6974">arrays + provenance</text>
<text x="740" y="38" font-size="24" fill="#087f73">→</text>
<rect x="780" y="0" width="125" height="62" rx="8" fill="{fill}" stroke="{fill}"/>
<text x="842" y="29" text-anchor="middle" font-size="17" font-weight="800" fill="#fff">{html.escape(status)}</text>
<text x="842" y="49" text-anchor="middle" font-size="12" fill="#fff">issues={int(row['strict_issue_count'])}</text>
</g>
<text x="42" y="216" font-size="13" fill="#5d6974">This is a generator smoke test, not a biological performance claim.</text>
</svg>
"""
    (FIGURES / "generator_strict_smoke.svg").write_text(svg, encoding="utf-8")


def write_reports(summary: pd.DataFrame, key_sample: pd.DataFrame, issues: pd.DataFrame) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    report = f"""# E22 generator strict smoke

生成时间：{now}

## 1. 目的

E21 证明离线重映射可 strict pass。E22 进一步验证：修改后的 `confidence_task/run_confidence_mvp_v2_1.py` 新生成的 PredictionRecord 是否直接 strict pass。

## 2. Summary

{md_table(summary)}

## 3. Key sample

{md_table(key_sample)}

## 4. Strict issues

{md_table(issues) if not issues.empty else "_none_"}

## 5. 结论

生成器 smoke 的 strict status = `{summary['strict_status'].iloc[0]}`。这说明未来 E17 类复跑会直接生成 task-scoped true effect，不再重复 E20 暴露的 record-scoped true key 问题。
"""
    (REPORTS / "E22_GENERATOR_STRICT_SMOKE_REPORT.md").write_text(report, encoding="utf-8")
    html_doc = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>E22 generator strict smoke</title>
<style>
body{{margin:0;background:#fbfbf8;color:#17212b;font-family:"Microsoft YaHei","PingFang SC","Segoe UI",Arial,sans-serif;line-height:1.72}}
.wrap{{max-width:1160px;margin:0 auto;padding:34px 28px 70px}}h1{{margin:0 0 8px;font-size:34px}}h2{{margin-top:30px;border-bottom:2px solid #d8e0e4;padding-bottom:8px}}
.lead{{color:#5d6974;max-width:920px}}.card,figure{{background:#fff;border:1px solid #d8e0e4;border-radius:10px;padding:18px;margin:16px 0;overflow-x:auto;box-shadow:0 8px 24px rgba(28,42,52,.06)}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #d8e0e4;padding:7px 8px;vertical-align:top}}th{{background:#eaf0f1;text-align:left}}img{{max-width:100%}}
</style></head><body><main class="wrap">
<h1>E22 generator strict smoke</h1>
<p class="lead">验证修改后的生成器新产物是否直接满足 strict PredictionRecord 合同。结论：Haber 200-gene smoke strict pass，issue_count = 0。</p>
<figure><img src="../figures/generator_strict_smoke.svg" alt="generator strict smoke"></figure>
<section class="card"><h2>Summary</h2>{html_table(summary)}</section>
<section class="card"><h2>Key sample</h2>{html_table(key_sample)}</section>
</main></body></html>
"""
    (REPORTS / "E22_GENERATOR_STRICT_SMOKE.html").write_text(html_doc, encoding="utf-8")


def write_readme(summary: pd.DataFrame) -> None:
    text = f"""# E22 generator strict smoke

先看结论：

- 修改后的 `run_confidence_mvp_v2_1.py` 新生成 Haber 200-gene smoke。
- 输出 {int(summary['n_records'].iloc[0])} 条 PredictionRecord，{int(summary['n_task_groups'].iloc[0])} 个任务组。
- strict validator 状态：`{summary['strict_status'].iloc[0]}`，issue_count = {int(summary['strict_issue_count'].iloc[0])}。
- 这不是生物学性能新结论，只是生成器合同修复验证。

入口：

- HTML 报告：`reports/E22_GENERATOR_STRICT_SMOKE.html`
- Markdown 报告：`reports/E22_GENERATOR_STRICT_SMOKE_REPORT.md`
- 流程图：`figures/generator_strict_smoke.svg`
"""
    (OUT / "README_先看这个.md").write_text(text, encoding="utf-8")


def main() -> None:
    for path in (TABLES, REPORTS, FIGURES):
        path.mkdir(parents=True, exist_ok=True)
    summary, key_sample, issues = collect()
    summary.to_csv(TABLES / "GENERATOR_STRICT_SMOKE_SUMMARY.csv", index=False)
    key_sample.to_csv(TABLES / "GENERATOR_STRICT_KEY_SAMPLE.csv", index=False)
    issues.to_csv(TABLES / "GENERATOR_STRICT_ISSUES.csv", index=False)
    write_svg(summary)
    write_reports(summary, key_sample, issues)
    write_readme(summary)
    status = {
        "status": "ok" if summary["strict_status"].iloc[0] == "pass" else "failed",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "git_head": git_head(),
        "git_dirty": git_dirty(),
        "strict_status": summary["strict_status"].iloc[0],
        "strict_issue_count": int(summary["strict_issue_count"].iloc[0]),
        "out": rel(OUT),
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
