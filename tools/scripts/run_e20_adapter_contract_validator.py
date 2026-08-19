#!/usr/bin/env python3
"""E20 package: audit whether predictor outputs satisfy the SafeConf adapter contract.

The script intentionally separates two questions:
1. Can an old bundle still be audited without losing evidence? (non-strict mode)
2. Is the bundle ready for future cross-predictor, same-task validation? (strict mode)
"""

from __future__ import annotations

import html
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = ROOT / "code" / "20260426_154505_perturb_transport_final_push"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from safetrans_confidence.data.records import (  # noqa: E402
    find_effect_array_files,
    validate_prediction_record_contract,
)


OUT = ROOT / "docs" / "实验结果" / "E20_adapter_contract_validator_20260707"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"
FIGURES = OUT / "figures"

GEARS_FORMAL_ROOT = Path("/home/yyf/safeconf_runtime/outputs/gears_prediction_records_formal")
E17_ROOT = ROOT / "runtime" / "e17_sciplex3_full743_gene5000_20260707"
SAFECONF_RUNTIME = Path("/home/yyf/safeconf_runtime")


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


def rel(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        try:
            return "$SAFECONF_RUNTIME/" + str(path.resolve().relative_to(SAFECONF_RUNTIME))
        except ValueError:
            return str(path)


def discover_bundles() -> list[dict[str, Any]]:
    bundles: list[dict[str, Any]] = [
        {
            "bundle_id": "E17_SCIPLEX3_FULL743_GENE5000",
            "bundle_group": "sciplex3_full743",
            "label": "sciplex3 full-743 gene5000 formal bundle",
            "run_dir": E17_ROOT,
            "records_csv": E17_ROOT / "input" / "PREDICTION_RECORDS.csv",
            "predicted_npz": E17_ROOT / "input" / "predicted_effects.npz",
            "true_npz": E17_ROOT / "input" / "true_effects.npz",
            "source_note": "Current strongest formal sciplex3 result package; task-level records use two baseline predictors.",
        }
    ]
    for records_csv in sorted(GEARS_FORMAL_ROOT.glob("*/seed_*/tables/PREDICTION_RECORDS.csv")):
        run_dir = records_csv.parents[1]
        dataset = run_dir.parent.name
        seed = run_dir.name
        bundles.append(
            {
                "bundle_id": f"GEARS_FORMAL_{dataset.upper()}_{seed.upper()}",
                "bundle_group": "gears_formal_legacy",
                "label": f"GEARS formal legacy bundle: {dataset}/{seed}",
                "run_dir": run_dir,
                "records_csv": records_csv,
                "predicted_npz": run_dir / "arrays" / "gears_predicted_effects.npz",
                "true_npz": run_dir / "arrays" / "gears_true_effects.npz",
                "source_note": "Existing GEARS vector output from earlier formal run; generated before strict provenance fields were added.",
            }
        )
    return bundles


def load_npz_meta(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"exists": False, "n_keys": 0, "size_mb": 0.0, "keys": set(), "sample_shape": "", "sample_dtype": ""}
    with np.load(path, mmap_mode="r") as arr:
        keys = list(arr.files)
        sample_shape = ""
        sample_dtype = ""
        if keys:
            sample = np.asarray(arr[keys[0]])
            sample_shape = "x".join(map(str, sample.shape))
            sample_dtype = str(sample.dtype)
    return {
        "exists": True,
        "n_keys": len(keys),
        "size_mb": path.stat().st_size / 1024 / 1024,
        "keys": set(keys),
        "sample_shape": sample_shape,
        "sample_dtype": sample_dtype,
    }


def sample_indices(n: int, max_n: int = 500) -> list[int]:
    if n <= max_n:
        return list(range(n))
    head = list(range(min(250, n)))
    tail = list(range(max(250, n - 250), n))
    return sorted(set(head + tail))


def issue_type(issue: str) -> str:
    for sep in ("=", ":"):
        if sep in issue:
            return issue.split(sep, 1)[0]
    return issue


def clean_issue(issue: str) -> str:
    return re.sub(r"np\.int64\(([-0-9]+)\)", r"\1", str(issue))


def audit_arrays(
    records: pd.DataFrame,
    predicted_path: Path | None,
    true_path: Path | None,
) -> tuple[list[str], dict[str, Any], list[dict[str, Any]]]:
    pred_meta = load_npz_meta(predicted_path)
    true_meta = load_npz_meta(true_path)
    array_rows = [
        {
            "array_role": "predicted_effects",
            "path": rel(predicted_path),
            "exists": pred_meta["exists"],
            "n_keys": pred_meta["n_keys"],
            "size_mb": pred_meta["size_mb"],
            "sample_shape": pred_meta["sample_shape"],
            "sample_dtype": pred_meta["sample_dtype"],
        },
        {
            "array_role": "true_effects",
            "path": rel(true_path),
            "exists": true_meta["exists"],
            "n_keys": true_meta["n_keys"],
            "size_mb": true_meta["size_mb"],
            "sample_shape": true_meta["sample_shape"],
            "sample_dtype": true_meta["sample_dtype"],
        },
    ]
    issues: list[str] = []
    if not pred_meta["exists"]:
        issues.append("missing_predicted_effect_array_file")
    if not true_meta["exists"]:
        issues.append("missing_true_effect_array_file")
    if not pred_meta["exists"] or not true_meta["exists"]:
        return issues, {"array_shape_checked_records": 0, "array_shape_check_mode": "not_available"}, array_rows

    wanted_pred = set(records.get("predicted_effect_key", pd.Series(dtype=str)).dropna().astype(str))
    wanted_true = set(records.get("true_effect_key", pd.Series(dtype=str)).dropna().astype(str))
    missing_pred = sorted(wanted_pred - pred_meta["keys"])
    missing_true = sorted(wanted_true - true_meta["keys"])
    if missing_pred:
        issues.append("missing_predicted_effect_arrays=" + ",".join(missing_pred[:5]))
    if missing_true:
        issues.append("missing_true_effect_arrays=" + ",".join(missing_true[:5]))

    checked = 0
    invalid_shape = 0
    mismatch = 0
    shape_values: list[int] = []
    rows = records.reset_index(drop=True)
    chosen = sample_indices(len(rows))
    with np.load(predicted_path, mmap_mode="r") as pred_arr, np.load(true_path, mmap_mode="r") as true_arr:
        for i in chosen:
            row = rows.iloc[i]
            pred_key = str(row.get("predicted_effect_key", ""))
            true_key = str(row.get("true_effect_key", ""))
            if pred_key not in pred_meta["keys"] or true_key not in true_meta["keys"]:
                continue
            pred = np.asarray(pred_arr[pred_key])
            true = np.asarray(true_arr[true_key])
            checked += 1
            if pred.ndim != 1 or true.ndim != 1 or pred.size == 0 or true.size == 0:
                invalid_shape += 1
                continue
            shape_values.append(int(pred.size))
            if pred.shape != true.shape:
                mismatch += 1
    if invalid_shape:
        issues.append(f"invalid_effect_array_shape_sampled={invalid_shape}")
    if mismatch:
        issues.append(f"effect_array_shape_mismatch_sampled={mismatch}")
    meta = {
        "array_shape_checked_records": checked,
        "array_shape_check_mode": "full" if len(records) <= len(chosen) else "sampled_head_tail",
        "array_shape_unique_gene_dims": ",".join(map(str, sorted(set(shape_values)))) if shape_values else "",
        "predicted_key_coverage": 1.0 - (len(missing_pred) / max(1, len(wanted_pred))),
        "true_key_coverage": 1.0 - (len(missing_true) / max(1, len(wanted_true))),
    }
    return issues, meta, array_rows


def audit_bundle(spec: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    issues_rows: list[dict[str, Any]] = []
    array_rows: list[dict[str, Any]] = []
    records_path = Path(spec["records_csv"])
    predicted_path = Path(spec["predicted_npz"])
    true_path = Path(spec["true_npz"])
    if not records_path.exists():
        issue = "missing_prediction_records_csv"
        return (
            {
                "bundle_id": spec["bundle_id"],
                "bundle_group": spec["bundle_group"],
                "label": spec["label"],
                "records_csv": rel(records_path),
                "predicted_npz": rel(predicted_path),
                "true_npz": rel(true_path),
                "exists": False,
                "strict_status": "fail",
                "non_strict_status": "fail",
                "strict_issue_count": 1,
                "non_strict_issue_count": 1,
                "primary_strict_issue": issue,
                "source_note": spec["source_note"],
            },
            [{"bundle_id": spec["bundle_id"], "mode": "strict", "issue_type": issue, "issue": issue}],
            array_rows,
        )

    records = pd.read_csv(records_path)
    record_strict = validate_prediction_record_contract(records, strict=True)
    record_non_strict = validate_prediction_record_contract(records, strict=False)
    array_issues, array_meta, array_rows = audit_arrays(records, predicted_path, true_path)
    strict_issues = record_strict + array_issues
    non_strict_issues = record_non_strict + array_issues

    for mode, mode_issues in [("strict", strict_issues), ("non_strict", non_strict_issues)]:
        for issue in mode_issues:
            issue = clean_issue(issue)
            issues_rows.append(
                {
                    "bundle_id": spec["bundle_id"],
                    "bundle_group": spec["bundle_group"],
                    "mode": mode,
                    "issue_type": issue_type(issue),
                    "issue": issue,
                }
            )
    for row in array_rows:
        row["bundle_id"] = spec["bundle_id"]
        row["bundle_group"] = spec["bundle_group"]

    issue_counter = Counter(issue_type(x) for x in strict_issues)
    primary = issue_counter.most_common(1)[0][0] if issue_counter else ""
    summary = {
        "bundle_id": spec["bundle_id"],
        "bundle_group": spec["bundle_group"],
        "label": spec["label"],
        "records_csv": rel(records_path),
        "predicted_npz": rel(predicted_path),
        "true_npz": rel(true_path),
        "exists": True,
        "n_records": int(len(records)),
        "n_tasks": int(records["task_key"].nunique()) if "task_key" in records.columns else 0,
        "n_predictors": int(records["predictor_name"].nunique()) if "predictor_name" in records.columns else 0,
        "predictors": ",".join(sorted(records["predictor_name"].dropna().astype(str).unique())) if "predictor_name" in records.columns else "",
        "n_datasets": int(records["dataset_name"].nunique()) if "dataset_name" in records.columns else 0,
        "dataset_names": ",".join(sorted(records["dataset_name"].dropna().astype(str).unique())) if "dataset_name" in records.columns else "",
        "n_folds": int(records["fold_id"].nunique()) if "fold_id" in records.columns else 0,
        "split_values": ",".join(sorted(records["split"].dropna().astype(str).unique())) if "split" in records.columns else "",
        "strict_status": "pass" if not strict_issues else "fail",
        "non_strict_status": "pass" if not non_strict_issues else "warn" if set(map(issue_type, non_strict_issues)) == {"legacy_true_effect_key_record_scoped_check_skipped"} else "fail",
        "strict_issue_count": int(len(strict_issues)),
        "non_strict_issue_count": int(len(non_strict_issues)),
        "primary_strict_issue": primary,
        "strict_issue_types": ";".join(f"{k}:{v}" for k, v in sorted(issue_counter.items())),
        "array_shape_checked_records": array_meta.get("array_shape_checked_records", 0),
        "array_shape_check_mode": array_meta.get("array_shape_check_mode", ""),
        "array_shape_unique_gene_dims": array_meta.get("array_shape_unique_gene_dims", ""),
        "predicted_key_coverage": array_meta.get("predicted_key_coverage", 0.0),
        "true_key_coverage": array_meta.get("true_key_coverage", 0.0),
        "source_note": spec["source_note"],
    }
    return summary, issues_rows, array_rows


def adapter_requirements(summary: pd.DataFrame) -> pd.DataFrame:
    e17 = summary[summary["bundle_id"].eq("E17_SCIPLEX3_FULL743_GENE5000")]
    gears = summary[summary["bundle_group"].eq("gears_formal_legacy")]
    e17_status = "present; strict fail because true_effect_key is record-scoped across predictors" if len(e17) else "not_found"
    gears_status = "legacy missing strict provenance columns in old outputs" if len(gears) else "not_found"
    return pd.DataFrame(
        [
            {
                "requirement": "PREDICTION_RECORDS.csv",
                "plain_meaning": "每一行是一条模型预测，至少要能说明数据集、任务、预测器、真实误差。",
                "why_it_matters": "没有这张表，SafeConf 无法把置信度分数和真实失败对应起来。",
                "current_e17_status": "present",
                "current_gears_legacy_status": "present",
                "future_action": "保留统一列名，不允许每个模型随意改字段。",
            },
            {
                "requirement": "predicted_effect_key / true_effect_key",
                "plain_meaning": "CSV 里的 key 必须能在 npz 里找到对应向量。",
                "why_it_matters": "这是从表格跳到真实基因表达向量的桥。",
                "current_e17_status": "present; key coverage audited",
                "current_gears_legacy_status": "present; key coverage audited",
                "future_action": "同一任务不同预测器应共享同一个 true_effect_key。",
            },
            {
                "requirement": "gene_panel_id + gene_order_hash",
                "plain_meaning": "说明用了哪些基因，以及这些基因的顺序。",
                "why_it_matters": "两个向量长度相同也可能基因顺序不同，不校验会产生假一致。",
                "current_e17_status": e17_status,
                "current_gears_legacy_status": gears_status,
                "future_action": "GEARS 导出器已补写；scGPT/CPA adapter 必须同样写出。",
            },
            {
                "requirement": "normalization_id + effect_definition",
                "plain_meaning": "说明 effect 是均值差、logFC 还是其它定义，归一化怎么做。",
                "why_it_matters": "不同 effect 定义不能直接混在一起比较 RMSE。",
                "current_e17_status": "present",
                "current_gears_legacy_status": "legacy missing in old outputs",
                "future_action": "统一使用 mean_diff 或明确转换。",
            },
            {
                "requirement": "strict contract pass",
                "plain_meaning": "能直接进入跨模型、同任务、同基因顺序的比较。",
                "why_it_matters": "这是把 GEARS、scGPT、CPA 放在一张表里比较的门槛。",
                "current_e17_status": e17_status,
                "current_gears_legacy_status": gears_status,
                "future_action": "先修 adapter，再重跑小规模三模型 smoke，最后扩展正式验证。",
            },
        ]
    )


def md_table(df: pd.DataFrame, cols: list[str] | None = None, n: int = 80) -> str:
    show = df if cols is None else df[cols]
    show = show.head(n).copy()
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


def html_table(df: pd.DataFrame, cols: list[str] | None = None, n: int = 80) -> str:
    show = df if cols is None else df[cols]
    return show.head(n).to_html(index=False, escape=True)


def write_svg(summary: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    groups = (
        summary.groupby("bundle_group")
        .agg(
            n_bundles=("bundle_id", "count"),
            strict_pass=("strict_status", lambda x: int((x == "pass").sum())),
            non_strict_ok=("non_strict_status", lambda x: int(x.isin(["pass", "warn"]).sum())),
        )
        .reset_index()
    )
    y = 82
    rows = []
    for _, row in groups.iterrows():
        n = max(1, int(row["n_bundles"]))
        strict_w = 360 * int(row["strict_pass"]) / n
        non_w = 360 * int(row["non_strict_ok"]) / n
        rows.append(f'<text x="44" y="{y+17}" font-size="16" font-weight="700" fill="#17212b">{html.escape(str(row["bundle_group"]))}</text>')
        rows.append(f'<rect x="300" y="{y}" width="360" height="18" rx="4" fill="#e8edef"/>')
        rows.append(f'<rect x="300" y="{y}" width="{strict_w:.1f}" height="18" rx="4" fill="#28734f"/>')
        rows.append(f'<text x="680" y="{y+15}" font-size="13" fill="#40515e">strict {int(row["strict_pass"])}/{n}</text>')
        rows.append(f'<rect x="300" y="{y+28}" width="360" height="18" rx="4" fill="#e8edef"/>')
        rows.append(f'<rect x="300" y="{y+28}" width="{non_w:.1f}" height="18" rx="4" fill="#087f73"/>')
        rows.append(f'<text x="680" y="{y+43}" font-size="13" fill="#40515e">usable {int(row["non_strict_ok"])}/{n}</text>')
        y += 82
    flow = """
<g transform="translate(44 270)">
<rect x="0" y="0" width="170" height="58" rx="8" fill="#ffffff" stroke="#d8e0e4"/>
<text x="85" y="25" text-anchor="middle" font-size="15" font-weight="700">Predictor</text>
<text x="85" y="45" text-anchor="middle" font-size="12" fill="#5d6974">GEARS / scGPT / CPA</text>
<text x="190" y="35" font-size="24" fill="#087f73">→</text>
<rect x="230" y="0" width="180" height="58" rx="8" fill="#ffffff" stroke="#d8e0e4"/>
<text x="320" y="25" text-anchor="middle" font-size="15" font-weight="700">PredictionRecord</text>
<text x="320" y="45" text-anchor="middle" font-size="12" fill="#5d6974">task + keys + provenance</text>
<text x="430" y="35" font-size="24" fill="#087f73">→</text>
<rect x="470" y="0" width="160" height="58" rx="8" fill="#ffffff" stroke="#d8e0e4"/>
<text x="550" y="25" text-anchor="middle" font-size="15" font-weight="700">Effect arrays</text>
<text x="550" y="45" text-anchor="middle" font-size="12" fill="#5d6974">predicted / true</text>
<text x="650" y="35" font-size="24" fill="#087f73">→</text>
<rect x="690" y="0" width="180" height="58" rx="8" fill="#ffffff" stroke="#d8e0e4"/>
<text x="780" y="25" text-anchor="middle" font-size="15" font-weight="700">SafeConf audit</text>
<text x="780" y="45" text-anchor="middle" font-size="12" fill="#5d6974">confidence vs error</text>
</g>
"""
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="380" viewBox="0 0 1000 380">
<rect width="1000" height="380" fill="#fbfbf8"/>
<text x="44" y="38" font-size="23" font-weight="800" fill="#17212b">E20 adapter contract validator</text>
<text x="44" y="61" font-size="14" fill="#5d6974">Strict pass means future cross-predictor validation is directly safe. Usable means old evidence can still be audited with stated caveats.</text>
{''.join(rows)}
{flow}
</svg>
"""
    (FIGURES / "adapter_contract_validator.svg").write_text(svg, encoding="utf-8")


def write_reports(summary: pd.DataFrame, issues: pd.DataFrame, arrays: pd.DataFrame, requirements: pd.DataFrame) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    key_cols = [
        "bundle_id",
        "bundle_group",
        "n_records",
        "n_tasks",
        "n_predictors",
        "strict_status",
        "non_strict_status",
        "primary_strict_issue",
        "array_shape_unique_gene_dims",
    ]
    issue_counts = (
        issues.groupby(["bundle_group", "mode", "issue_type"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["bundle_group", "mode", "n"], ascending=[True, True, False])
        if not issues.empty
        else pd.DataFrame(columns=["bundle_group", "mode", "issue_type", "n"])
    )
    report = f"""# E20 adapter contract validator

生成时间：{now}

## 1. 这次做了什么

E20 检查当前已有预测输出是否满足 SafeConf 统一适配器合同。这个合同主要约束三件事：CSV 任务表、预测/真实 effect 向量、基因顺序与归一化说明。

## 2. 总结

{md_table(summary, key_cols)}

## 3. 问题类型统计

{md_table(issue_counts)}

## 4. 适配器要求

{md_table(requirements)}

## 5. 结论口径

- E17 sciplex3 full-743 gene5000 结果包可继续作为当前 strongest formal package 使用；它的数组 key coverage 正常。
- E17 在严格跨模型合同下仍有一个老问题：同一任务下不同预测器使用 record-scoped true_effect_key。后续三模型统一验证前要改成 task-scoped true_effect_key。
- 旧 GEARS formal 输出能做 non-strict 审计；严格模式失败的主因是旧 CSV 缺少 gene_panel_id、gene_order_hash、normalization_id 等来源字段。
- GEARS 导出脚本已经补写这些严格字段。以后新跑 GEARS 时，输出会更接近 strict contract。
"""
    (REPORTS / "E20_ADAPTER_CONTRACT_VALIDATOR_REPORT.md").write_text(report, encoding="utf-8")

    html_doc = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>E20 adapter contract validator</title>
<style>
body{{margin:0;background:#fbfbf8;color:#17212b;font-family:"Microsoft YaHei","PingFang SC","Segoe UI",Arial,sans-serif;line-height:1.72}}
.wrap{{max-width:1180px;margin:0 auto;padding:34px 28px 70px}}h1{{margin:0 0 8px;font-size:34px}}h2{{margin-top:30px;border-bottom:2px solid #d8e0e4;padding-bottom:8px}}
.lead{{color:#5d6974;max-width:940px}}.card,figure{{background:#fff;border:1px solid #d8e0e4;border-radius:10px;padding:18px;margin:16px 0;overflow-x:auto;box-shadow:0 8px 24px rgba(28,42,52,.06)}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #d8e0e4;padding:7px 8px;vertical-align:top}}th{{background:#eaf0f1;text-align:left}}
.ok{{color:#28734f;font-weight:700}}.warn{{color:#946205;font-weight:700}}.fail{{color:#a4403a;font-weight:700}}img{{max-width:100%}}code{{background:#f4f7f7;border:1px solid #dce4e7;border-radius:4px;padding:2px 5px}}
</style></head><body><main class="wrap">
<h1>E20 adapter contract validator</h1>
<p class="lead">这页只回答一个工程问题：现有预测输出能不能安全进入后续跨模型验证。结论很清楚：旧结果可以审计，但严格适配器还要修；GEARS 导出器已经开始补齐严格字段。</p>
<figure><img src="../figures/adapter_contract_validator.svg" alt="adapter contract validator"></figure>
<section class="card"><h2>1. Bundle summary</h2>{html_table(summary, key_cols)}</section>
<section class="card"><h2>2. Issue type counts</h2>{html_table(issue_counts)}</section>
<section class="card"><h2>3. Adapter requirements</h2>{html_table(requirements)}</section>
<section class="card"><h2>4. Array files</h2>{html_table(arrays, n=40)}</section>
<section class="card"><h2>5. Plain conclusion</h2>
<p>E17 的正式 sciplex3 包是可用证据；旧 GEARS 包是可审计证据；下一步如果要冲更强文章，必须做 task-scoped true effect、统一 gene order、统一 normalization 的三模型 adapter。</p>
</section>
</main></body></html>
"""
    (REPORTS / "E20_ADAPTER_CONTRACT_VALIDATOR.html").write_text(html_doc, encoding="utf-8")


def write_readme(summary: pd.DataFrame, requirements: pd.DataFrame) -> None:
    readme = f"""# E20 adapter contract validator

先看结论：

- 这不是新模型结果，而是一次“能不能安全接真实预测器”的源码级体检。
- E17 sciplex3 full-743 gene5000 包继续可用；数组 key coverage 正常。
- 旧 GEARS 输出能 non-strict 审计，但 strict 合同失败，主要因为旧记录缺少 gene order / normalization 等字段。
- GEARS 导出器已补写严格字段，后续重跑会更干净。

入口：

- HTML 报告：`reports/E20_ADAPTER_CONTRACT_VALIDATOR.html`
- Markdown 报告：`reports/E20_ADAPTER_CONTRACT_VALIDATOR_REPORT.md`
- 合同流程图：`figures/adapter_contract_validator.svg`
- 汇总表：`tables/ADAPTER_CONTRACT_BUNDLE_SUMMARY.csv`
- 问题明细：`tables/ADAPTER_CONTRACT_ISSUES.csv`

核心表：

{md_table(summary, ["bundle_id", "bundle_group", "n_records", "strict_status", "non_strict_status", "primary_strict_issue"], n=20)}

适配器合同：

{md_table(requirements, ["requirement", "plain_meaning", "future_action"])}
"""
    (OUT / "README_先看这个.md").write_text(readme, encoding="utf-8")


def main() -> None:
    for path in (TABLES, REPORTS, FIGURES):
        path.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    issue_rows: list[dict[str, Any]] = []
    array_rows: list[dict[str, Any]] = []
    for spec in discover_bundles():
        summary, issues, arrays = audit_bundle(spec)
        summaries.append(summary)
        issue_rows.extend(issues)
        array_rows.extend(arrays)

    summary_df = pd.DataFrame(summaries)
    issues_df = pd.DataFrame(issue_rows)
    arrays_df = pd.DataFrame(array_rows)
    requirements_df = adapter_requirements(summary_df)

    summary_df.to_csv(TABLES / "ADAPTER_CONTRACT_BUNDLE_SUMMARY.csv", index=False)
    issues_df.to_csv(TABLES / "ADAPTER_CONTRACT_ISSUES.csv", index=False)
    arrays_df.to_csv(TABLES / "ADAPTER_CONTRACT_ARRAY_SUMMARY.csv", index=False)
    requirements_df.to_csv(TABLES / "ADAPTER_REQUIREMENTS.csv", index=False)
    write_svg(summary_df)
    write_reports(summary_df, issues_df, arrays_df, requirements_df)
    write_readme(summary_df, requirements_df)
    status = {
        "status": "ok",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "git_head": git_head(),
        "git_dirty": git_dirty(),
        "n_bundles": int(len(summary_df)),
        "n_strict_pass": int(summary_df["strict_status"].eq("pass").sum()),
        "n_non_strict_usable": int(summary_df["non_strict_status"].isin(["pass", "warn"]).sum()),
        "out": rel(OUT),
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
