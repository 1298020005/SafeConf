#!/usr/bin/env python3
"""E21 package: prove the task-scoped true-effect remediation can strict-pass.

This script does not rewrite the E17 formal result. It builds a small,
auditable sample bundle from E17 with task-scoped true_effect_key values, then
validates the sample with the strict PredictionRecord contract.
"""

from __future__ import annotations

import html
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = ROOT / "code" / "20260426_154505_perturb_transport_final_push"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from safetrans_confidence.data.records import validate_prediction_record_contract  # noqa: E402


SOURCE = ROOT / "runtime" / "e17_sciplex3_full743_gene5000_20260707"
OUT = ROOT / "docs" / "实验结果" / "E21_strict_contract_remediation_20260707"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"
FIGURES = OUT / "figures"
INPUT = OUT / "input"


GROUP_COLS = ["dataset_name", "fold_id", "split", "task_key", "context", "perturbation"]


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


def safe_piece(value: object) -> str:
    text = str(value)
    keep = []
    for ch in text:
        keep.append(ch if ch.isalnum() or ch in {"-", "_", "."} else "_")
    return "".join(keep).strip("_")[:90] or "na"


def task_scoped_true_key(row: pd.Series) -> str:
    return (
        f"{safe_piece(row['dataset_name'])}::fold{int(row['fold_id'])}::"
        f"{safe_piece(row['split'])}::{safe_piece(row['task_key'])}::true_effect"
    )


def task_scoped_control_key(row: pd.Series) -> str:
    return (
        f"{safe_piece(row['dataset_name'])}::fold{int(row['fold_id'])}::"
        f"{safe_piece(row['split'])}::{safe_piece(row['task_key'])}::target_control_mean"
    )


def choose_groups(records: pd.DataFrame, per_split: int = 10) -> pd.DataFrame:
    groups = (
        records.groupby(GROUP_COLS, dropna=False)
        .agg(n_predictors=("predictor_name", "nunique"), n_records=("record_id", "count"))
        .reset_index()
    )
    groups = groups[groups["n_predictors"].ge(2)].copy()
    chosen = []
    for split in ["train", "val", "test"]:
        part = groups[groups["split"].astype(str).eq(split)].head(per_split)
        chosen.append(part)
    out = pd.concat(chosen, ignore_index=True)
    return out[GROUP_COLS]


def build_sample() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    records_path = SOURCE / "input" / "PREDICTION_RECORDS.csv"
    predicted_path = SOURCE / "input" / "predicted_effects.npz"
    true_path = SOURCE / "input" / "true_effects.npz"
    control_path = SOURCE / "input" / "target_control_means.npz"
    records = pd.read_csv(records_path)
    groups = choose_groups(records)
    sample = records.merge(groups, on=GROUP_COLS, how="inner").copy()
    sample = sample.sort_values(GROUP_COLS + ["predictor_name"]).reset_index(drop=True)

    pred_arrays: dict[str, np.ndarray] = {}
    true_arrays: dict[str, np.ndarray] = {}
    control_arrays: dict[str, np.ndarray] = {}
    remap_rows: list[dict[str, Any]] = []
    strict_rows = []
    true_diffs = []
    with np.load(predicted_path) as pred_npz, np.load(true_path) as true_npz, np.load(control_path) as ctrl_npz:
        for group_key, group in sample.groupby(GROUP_COLS, dropna=False):
            first = group.iloc[0]
            new_true_key = task_scoped_true_key(first)
            new_ctrl_key = task_scoped_control_key(first)
            true_vectors = [np.asarray(true_npz[str(key)], dtype=np.float32) for key in group["true_effect_key"]]
            ctrl_vectors = [np.asarray(ctrl_npz[str(key)], dtype=np.float32) for key in group["target_control_key"]]
            if len(true_vectors) > 1:
                true_diffs.append(float(max(np.max(np.abs(v - true_vectors[0])) for v in true_vectors[1:])))
            else:
                true_diffs.append(0.0)
            true_arrays[new_true_key] = true_vectors[0]
            control_arrays[new_ctrl_key] = ctrl_vectors[0]
            for _, row in group.iterrows():
                new_record_id = f"e21_rec_{len(strict_rows):05d}"
                new_pred_key = f"{new_record_id}::predicted_effect"
                pred_arrays[new_pred_key] = np.asarray(pred_npz[str(row["predicted_effect_key"])], dtype=np.float32)
                fixed = row.copy()
                fixed["record_id"] = new_record_id
                fixed["predicted_effect_key"] = new_pred_key
                fixed["true_effect_key"] = new_true_key
                fixed["target_control_key"] = new_ctrl_key
                strict_rows.append(fixed.to_dict())
                remap_rows.append(
                    {
                        "old_record_id": row["record_id"],
                        "new_record_id": new_record_id,
                        "predictor_name": row["predictor_name"],
                        "task_key": row["task_key"],
                        "split": row["split"],
                        "old_predicted_effect_key": row["predicted_effect_key"],
                        "new_predicted_effect_key": new_pred_key,
                        "old_true_effect_key": row["true_effect_key"],
                        "new_true_effect_key": new_true_key,
                    }
                )

    strict_records = pd.DataFrame(strict_rows)
    issues = validate_prediction_record_contract(
        strict_records,
        strict=True,
        predicted_effects=pred_arrays,
        true_effects=true_arrays,
    )
    INPUT.mkdir(parents=True, exist_ok=True)
    strict_records.to_csv(INPUT / "PREDICTION_RECORDS.csv", index=False)
    np.savez_compressed(INPUT / "predicted_effects.npz", **pred_arrays)
    np.savez_compressed(INPUT / "true_effects.npz", **true_arrays)
    np.savez_compressed(INPUT / "target_control_means.npz", **control_arrays)
    remap = pd.DataFrame(remap_rows)
    summary = pd.DataFrame(
        [
            {
                "source_bundle": rel(SOURCE),
                "source_records": int(len(records)),
                "sample_records": int(len(strict_records)),
                "sample_task_groups": int(len(groups)),
                "sample_predictors": ",".join(sorted(strict_records["predictor_name"].dropna().astype(str).unique())),
                "sample_splits": ",".join(sorted(strict_records["split"].dropna().astype(str).unique())),
                "strict_status": "pass" if not issues else "fail",
                "strict_issue_count": int(len(issues)),
                "strict_issues": ";".join(map(str, issues)),
                "max_true_effect_diff_within_task": float(max(true_diffs) if true_diffs else 0.0),
                "n_predicted_arrays": int(len(pred_arrays)),
                "n_true_arrays": int(len(true_arrays)),
                "n_control_arrays": int(len(control_arrays)),
            }
        ]
    )
    return strict_records, remap, summary, issues


def md_table(df: pd.DataFrame, n: int = 80) -> str:
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


def html_table(df: pd.DataFrame, n: int = 80) -> str:
    return df.head(n).to_html(index=False, escape=True)


def write_svg(summary: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    row = summary.iloc[0]
    strict_fill = "#28734f" if row["strict_status"] == "pass" else "#a4403a"
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="980" height="280" viewBox="0 0 980 280">
<rect width="980" height="280" fill="#fbfbf8"/>
<text x="42" y="38" font-size="23" font-weight="800" fill="#17212b">E21 strict contract remediation smoke</text>
<text x="42" y="62" font-size="14" fill="#5d6974">Task-scoped true_effect_key turns the sampled two-predictor E17 bundle into a strict-pass adapter example.</text>
<g transform="translate(42 98)">
<rect x="0" y="0" width="170" height="62" rx="8" fill="#fff" stroke="#d8e0e4"/>
<text x="85" y="25" text-anchor="middle" font-size="15" font-weight="700">E17 legacy sample</text>
<text x="85" y="46" text-anchor="middle" font-size="12" fill="#5d6974">{int(row['sample_records'])} records</text>
<text x="190" y="38" font-size="24" fill="#087f73">→</text>
<rect x="230" y="0" width="210" height="62" rx="8" fill="#fff" stroke="#d8e0e4"/>
<text x="335" y="25" text-anchor="middle" font-size="15" font-weight="700">Remap true keys</text>
<text x="335" y="46" text-anchor="middle" font-size="12" fill="#5d6974">{int(row['sample_task_groups'])} task-scoped true arrays</text>
<text x="460" y="38" font-size="24" fill="#087f73">→</text>
<rect x="500" y="0" width="200" height="62" rx="8" fill="#fff" stroke="#d8e0e4"/>
<text x="600" y="25" text-anchor="middle" font-size="15" font-weight="700">Strict validator</text>
<text x="600" y="46" text-anchor="middle" font-size="12" fill="#5d6974">same gene order + task truth</text>
<text x="720" y="38" font-size="24" fill="#087f73">→</text>
<rect x="760" y="0" width="145" height="62" rx="8" fill="{strict_fill}" stroke="{strict_fill}"/>
<text x="832" y="28" text-anchor="middle" font-size="17" font-weight="800" fill="#fff">STRICT {html.escape(str(row['strict_status']).upper())}</text>
<text x="832" y="48" text-anchor="middle" font-size="12" fill="#fff">issues={int(row['strict_issue_count'])}</text>
</g>
<text x="42" y="220" font-size="13" fill="#5d6974">This is a remediation smoke, not a replacement for the E17 formal evidence package.</text>
</svg>
"""
    (FIGURES / "strict_contract_remediation_smoke.svg").write_text(svg, encoding="utf-8")


def write_reports(strict_records: pd.DataFrame, remap: pd.DataFrame, summary: pd.DataFrame) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    report = f"""# E21 strict contract remediation smoke

生成时间：{now}

## 1. 目的

E20 发现 E17 类双预测器记录的主要 strict 问题是：同一任务下不同预测器使用 record-scoped `true_effect_key`。E21 不改 E17 正式结果，只抽样构造一个 task-scoped true effect 小包，证明修法可以通过 strict contract。

## 2. Summary

{md_table(summary)}

## 3. Key remap sample

{md_table(remap.head(20))}

## 4. 结论

- 抽样包 strict status = `{summary['strict_status'].iloc[0]}`。
- 同一任务内 true effect 最大差异 = `{summary['max_true_effect_diff_within_task'].iloc[0]}`，说明把 true key 合并到任务级是合理的。
- 这不是 E17 正式结果替换，只是为下一轮 full rerun / shared benchmark adapter 提供可复用修法。
"""
    (REPORTS / "E21_STRICT_CONTRACT_REMEDIATION_REPORT.md").write_text(report, encoding="utf-8")
    html_doc = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>E21 strict contract remediation</title>
<style>
body{{margin:0;background:#fbfbf8;color:#17212b;font-family:"Microsoft YaHei","PingFang SC","Segoe UI",Arial,sans-serif;line-height:1.72}}
.wrap{{max-width:1160px;margin:0 auto;padding:34px 28px 70px}}h1{{margin:0 0 8px;font-size:34px}}h2{{margin-top:30px;border-bottom:2px solid #d8e0e4;padding-bottom:8px}}
.lead{{color:#5d6974;max-width:920px}}.card,figure{{background:#fff;border:1px solid #d8e0e4;border-radius:10px;padding:18px;margin:16px 0;overflow-x:auto;box-shadow:0 8px 24px rgba(28,42,52,.06)}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #d8e0e4;padding:7px 8px;vertical-align:top}}th{{background:#eaf0f1;text-align:left}}img{{max-width:100%}}
</style></head><body><main class="wrap">
<h1>E21 strict contract remediation smoke</h1>
<p class="lead">E21 把 E20 暴露出的 true_effect_key 问题变成一个可验证修法：同一任务下不同预测器共享 task-scoped true effect，抽样包 strict pass。</p>
<figure><img src="../figures/strict_contract_remediation_smoke.svg" alt="strict remediation smoke"></figure>
<section class="card"><h2>Summary</h2>{html_table(summary)}</section>
<section class="card"><h2>Key remap sample</h2>{html_table(remap, n=30)}</section>
<section class="card"><h2>Output bundle</h2><p>小型 strict bundle 位于 <code>input/PREDICTION_RECORDS.csv</code>、<code>input/predicted_effects.npz</code>、<code>input/true_effects.npz</code>。</p></section>
</main></body></html>
"""
    (REPORTS / "E21_STRICT_CONTRACT_REMEDIATION.html").write_text(html_doc, encoding="utf-8")


def write_readme(summary: pd.DataFrame) -> None:
    text = f"""# E21 strict contract remediation smoke

先看结论：

- E21 不替换 E17 正式结果，只做一个小型 strict contract 修法验证。
- 从 E17 抽样 {int(summary['sample_records'].iloc[0])} 条记录、{int(summary['sample_task_groups'].iloc[0])} 个任务组。
- 将 `true_effect_key` 改为 task-scoped 后，strict validator 状态：`{summary['strict_status'].iloc[0]}`。
- 这说明下一轮 full rerun / shared benchmark adapter 应采用 task-scoped true effect。

入口：

- HTML 报告：`reports/E21_STRICT_CONTRACT_REMEDIATION.html`
- Markdown 报告：`reports/E21_STRICT_CONTRACT_REMEDIATION_REPORT.md`
- 流程图：`figures/strict_contract_remediation_smoke.svg`
- 小型 strict bundle：`input/PREDICTION_RECORDS.csv`
"""
    (OUT / "README_先看这个.md").write_text(text, encoding="utf-8")


def main() -> None:
    for path in (TABLES, REPORTS, FIGURES, INPUT):
        path.mkdir(parents=True, exist_ok=True)
    strict_records, remap, summary, issues = build_sample()
    strict_records.to_csv(TABLES / "STRICT_SAMPLE_PREDICTION_RECORDS.csv", index=False)
    remap.to_csv(TABLES / "STRICT_KEY_REMAP.csv", index=False)
    summary.to_csv(TABLES / "STRICT_REMEDIATION_SUMMARY.csv", index=False)
    pd.DataFrame({"issue": list(map(str, issues))}).to_csv(TABLES / "STRICT_VALIDATION_ISSUES.csv", index=False)
    write_svg(summary)
    write_reports(strict_records, remap, summary)
    write_readme(summary)
    status = {
        "status": "ok" if not issues else "failed",
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
