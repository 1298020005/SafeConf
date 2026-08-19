#!/usr/bin/env python3
"""E24 package: model-family compatibility audit for adapter work.

E23 freezes a strict PredictionRecord contract smoke using Haber stimuli.
E24 audits whether that manifest is biologically and technically compatible
with GEARS, scGPT and CPA/chemCPA, and identifies the next executable manifest.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "实验结果" / "E24_model_family_compatibility_audit_20260708"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"
FIGURES = OUT / "figures"

E23_MANIFEST = ROOT / "docs" / "实验结果" / "E23_shared_benchmark_adapter_workbench_20260708" / "input" / "SHARED_BENCHMARK_TASK_MANIFEST.csv"
GEARS_FORMAL = Path("/home/yyf/safeconf_runtime/outputs/gears_prediction_records_formal")
GEARS_DATA = Path("/home/yyf/data/gears_formal_baselines_v2")
SCGPT_ENV = Path("/home/yyf/.conda/envs/scgpt_env")
SCGPT_ZIP = Path("/home/yyf/archive/code/20260519_0958_home_cleanup/moved_top_level/scGPT-main.zip")
CPA_PDF = Path("/home/yyf/safeconf_runtime/outputs/current_project_explanation_20260530/合并版_给我看的/papers/CPA_2023_MSB.pdf")


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
        if str(path).startswith("/home/yyf/safeconf_runtime"):
            return "$SAFECONF_RUNTIME" + str(path)[len("/home/yyf/safeconf_runtime") :]
        if str(path).startswith("/home/yyf/data"):
            return "$YYF_DATA" + str(path)[len("/home/yyf/data") :]
        if str(path).startswith("/home/yyf/.conda"):
            return "$YYF_CONDA" + str(path)[len("/home/yyf/.conda") :]
        if str(path).startswith("/home/yyf/archive"):
            return "$YYF_ARCHIVE" + str(path)[len("/home/yyf/archive") :]
        return str(path)


def module_importable(py: Path, module: str) -> bool:
    if not py.exists():
        return False
    code = f"import importlib.util; print(bool(importlib.util.find_spec('{module}')))"
    proc = subprocess.run(
        [str(py), "-c", code],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=20,
    )
    return proc.stdout.strip().lower() == "true"


def classify_perturbation(value: object) -> str:
    text = str(value)
    low = text.lower()
    if "+" in text or low.endswith("_ctrl") or low.endswith("+ctrl"):
        return "gene_perturbation_like"
    if any(token in low for token in ["day", "salmonella", "hpoly", "lps", "ifn", "tnf"]):
        return "stimulus_or_timecourse"
    if any(ch.isdigit() for ch in low) and ("_" in text or "-" in text):
        return "drug_or_dose_like"
    return "unknown_or_contextual"


def audit_e23_manifest() -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(E23_MANIFEST)
    class_counts = (
        manifest.assign(perturbation_class=manifest["perturbation"].map(classify_perturbation))
        .groupby("perturbation_class", dropna=False)
        .size()
        .reset_index(name="n_task_groups")
    )
    rows = [
        {
            "manifest": "E23_HABER_200GENE_STRICT_SMOKE",
            "n_task_groups": len(manifest),
            "dataset_names": ",".join(sorted(manifest["dataset_name"].astype(str).unique())),
            "perturbation_examples": ",".join(sorted(manifest["perturbation"].astype(str).unique())[:10]),
            "gene_order_hashes": int(manifest["gene_order_hash"].nunique()),
            "effect_definitions": ",".join(sorted(manifest["effect_definition"].astype(str).unique())),
            "gears_compatible": False,
            "gears_reason": "Perturbations are Hpoly/Salmonella stimuli, not gene knockout/overexpression conditions.",
            "scgpt_compatible": "adapter_possible_but_not_ready",
            "scgpt_reason": "Generic single-cell model source exists, but perturbation prediction adapter/checkpoint is absent.",
            "cpa_compatible": "conceptually_possible_for_stimulus_or_drug_but_not_ready",
            "cpa_reason": "CPA/chemCPA assets are not locally executable; E23 stimuli are not a prepared CPA dataset.",
        }
    ]
    return pd.DataFrame(rows), class_counts


def collect_gears_candidate_records() -> pd.DataFrame:
    rows = []
    for records_csv in sorted(GEARS_FORMAL.glob("*/seed_*/tables/PREDICTION_RECORDS.csv")):
        rec = pd.read_csv(records_csv)
        rec["source_records_csv"] = rel(records_csv)
        rec["source_run_dir"] = rel(records_csv.parents[1])
        rows.append(rec)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["perturbation_class"] = out["perturbation"].map(classify_perturbation)
    return out


def build_gears_candidate_summary(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if records.empty:
        return pd.DataFrame(), pd.DataFrame()
    summary = (
        records.groupby("dataset_name", dropna=False)
        .agg(
            n_records=("record_id", "count"),
            n_task_like_rows=("task_key", "nunique"),
            n_seeds=("fold_id", "nunique"),
            n_perturbations=("perturbation", "nunique"),
            perturbation_class=("perturbation_class", lambda x: ",".join(sorted(set(map(str, x))))),
            mean_rmse=("true_error_rmse", "mean"),
        )
        .reset_index()
    )
    manifest_cols = [
        "dataset_name",
        "fold_id",
        "split",
        "task_key",
        "context",
        "perturbation",
        "perturbation_class",
        "source_records_csv",
    ]
    candidate_manifest = records[manifest_cols].drop_duplicates().sort_values(
        ["dataset_name", "fold_id", "task_key"]
    )
    return summary, candidate_manifest


def inspect_processed_h5ad_inventory() -> pd.DataFrame:
    rows = []
    for path in sorted(GEARS_DATA.glob("*_local_atlas/perturb_processed.h5ad")):
        dataset = path.parent.name.replace("_local_atlas", "")
        rows.append(
            {
                "dataset": dataset,
                "processed_h5ad": rel(path),
                "exists": path.exists(),
                "size_mb": path.stat().st_size / 1024 / 1024 if path.exists() else 0,
                "has_split_files": (path.parent / "splits").exists(),
                "n_split_files": len(list((path.parent / "splits").glob("*.pkl"))) if (path.parent / "splits").exists() else 0,
            }
        )
    return pd.DataFrame(rows)


def build_environment_table() -> pd.DataFrame:
    py = SCGPT_ENV / "bin" / "python"
    return pd.DataFrame(
        [
            {
                "environment": "default_python",
                "path": sys.executable,
                "gears_importable": module_importable(Path(sys.executable), "gears"),
                "scgpt_importable": module_importable(Path(sys.executable), "scgpt"),
            },
            {
                "environment": "scgpt_env",
                "path": rel(py),
                "gears_importable": module_importable(py, "gears"),
                "scgpt_importable": module_importable(py, "scgpt"),
            },
        ]
    )


def build_decision_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "decision": "Do not run GEARS on E23 Haber manifest",
                "reason": "E23 perturbations are stimulus/timecourse labels, not gene perturbations.",
                "action": "Keep E23 as adapter contract smoke only.",
                "priority": 1,
            },
            {
                "decision": "Use GEARS-compatible gene perturbation data for the first true model adapter",
                "reason": "Norman/Adamson/Dixit/Frangieh local processed GEARS assets exist and scgpt_env imports gears.",
                "action": "Run a small GEARS strict smoke in scgpt_env, then package as E25 if runtime succeeds.",
                "priority": 2,
            },
            {
                "decision": "Do not claim scGPT validation yet",
                "reason": "scGPT source zip exists but import is false and no perturbation checkpoint/output was found.",
                "action": "Unpack/install source or point PYTHONPATH; then locate checkpoint/tutorial assets.",
                "priority": 3,
            },
            {
                "decision": "Do not claim CPA/chemCPA validation yet",
                "reason": "Only a CPA PDF trace was found locally.",
                "action": "Acquire executable assets or choose an accessible perturbation model.",
                "priority": 4,
            },
        ]
    )


def md_table(df: pd.DataFrame, n: int = 80) -> str:
    if df.empty:
        return "_empty_"
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
    if df.empty:
        return "<p><em>empty</em></p>"
    return df.head(n).to_html(index=False, escape=True)


def write_svg(e23_summary: pd.DataFrame, env: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    gears_env = bool(env.loc[env["environment"].eq("scgpt_env"), "gears_importable"].iloc[0]) if not env.empty else False
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1020" height="320" viewBox="0 0 1020 320">
<rect width="1020" height="320" fill="#fbfbf8"/>
<text x="42" y="38" font-size="23" font-weight="800" fill="#17212b">E24 model-family compatibility audit</text>
<text x="42" y="62" font-size="14" fill="#5d6974">Contract validity and model-family compatibility are different gates. E23 passes the first gate, not the GEARS biology gate.</text>
<g transform="translate(48 104)">
<rect x="0" y="0" width="220" height="70" rx="8" fill="#fff3d6" stroke="#ead89a"/>
<text x="110" y="26" text-anchor="middle" font-size="15" font-weight="700" fill="#946205">E23 Haber manifest</text>
<text x="110" y="49" text-anchor="middle" font-size="12" fill="#946205">stimulus/timecourse perturbations</text>
<text x="245" y="42" font-size="24" fill="#087f73">→</text>
<rect x="285" y="0" width="220" height="70" rx="8" fill="#faeae8" stroke="#e8c4c0"/>
<text x="395" y="26" text-anchor="middle" font-size="15" font-weight="700" fill="#a4403a">GEARS compatibility</text>
<text x="395" y="49" text-anchor="middle" font-size="12" fill="#a4403a">not suitable for direct GEARS run</text>
<text x="530" y="42" font-size="24" fill="#087f73">→</text>
<rect x="570" y="0" width="260" height="70" rx="8" fill="#e7f3eb" stroke="#c8dfd0"/>
<text x="700" y="26" text-anchor="middle" font-size="15" font-weight="700" fill="#28734f">Next executable route</text>
<text x="700" y="49" text-anchor="middle" font-size="12" fill="#28734f">GEARS gene-perturbation smoke</text>
</g>
<g transform="translate(48 230)">
<rect x="0" y="0" width="360" height="45" rx="8" fill="#fff" stroke="#d8e0e4"/>
<text x="20" y="28" font-size="14" fill="#17212b">scgpt_env imports GEARS: {str(gears_env)}</text>
<rect x="430" y="0" width="390" height="45" rx="8" fill="#fff" stroke="#d8e0e4"/>
<text x="450" y="28" font-size="14" fill="#17212b">E23 task groups: {int(e23_summary['n_task_groups'].iloc[0])}</text>
</g>
</svg>
"""
    (FIGURES / "model_family_compatibility_audit.svg").write_text(svg, encoding="utf-8")


def write_reports(
    e23_summary: pd.DataFrame,
    class_counts: pd.DataFrame,
    gears_summary: pd.DataFrame,
    candidate_manifest: pd.DataFrame,
    h5ad_inventory: pd.DataFrame,
    env: pd.DataFrame,
    decisions: pd.DataFrame,
) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    report = f"""# E24 model-family compatibility audit

生成时间：{now}

## 1. 目的

E23 已经建立 strict adapter contract smoke，但它来自 Haber stimulus/timecourse 数据。E24 检查这个 manifest 是否适合直接给 GEARS、scGPT、CPA/chemCPA 使用。

## 2. E23 compatibility summary

{md_table(e23_summary)}

## 3. E23 perturbation class

{md_table(class_counts)}

## 4. GEARS candidate legacy records

{md_table(gears_summary)}

## 5. GEARS processed assets

{md_table(h5ad_inventory)}

## 6. Runtime environments

{md_table(env)}

## 7. Decisions

{md_table(decisions)}

## 8. 结论

- 不应把 E23 Haber stimulus manifest 直接作为 GEARS biological benchmark。
- GEARS 的可执行路径存在：`scgpt_env` 可以 import GEARS，且本地有 Norman/Adamson/Dixit/Frangieh processed assets。
- 下一步应做 GEARS gene-perturbation strict smoke，而不是 GEARS-on-E23。
- scGPT/CPA 仍需先解决安装、checkpoint 或可执行输出资产。
"""
    (REPORTS / "E24_MODEL_FAMILY_COMPATIBILITY_AUDIT_REPORT.md").write_text(report, encoding="utf-8")
    html_doc = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>E24 model-family compatibility audit</title>
<style>
body{{margin:0;background:#fbfbf8;color:#17212b;font-family:"Microsoft YaHei","PingFang SC","Segoe UI",Arial,sans-serif;line-height:1.72}}
.wrap{{max-width:1180px;margin:0 auto;padding:34px 28px 70px}}h1{{margin:0 0 8px;font-size:34px}}h2{{margin-top:30px;border-bottom:2px solid #d8e0e4;padding-bottom:8px}}
.lead{{color:#5d6974;max-width:940px}}.card,figure{{background:#fff;border:1px solid #d8e0e4;border-radius:10px;padding:18px;margin:16px 0;overflow-x:auto;box-shadow:0 8px 24px rgba(28,42,52,.06)}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #d8e0e4;padding:7px 8px;vertical-align:top}}th{{background:#eaf0f1;text-align:left}}img{{max-width:100%}}
</style></head><body><main class="wrap">
<h1>E24 model-family compatibility audit</h1>
<p class="lead">E24 把一个容易踩坑的区别说清楚：PredictionRecord 合同过关，不代表某个模型家族可以生物学合理地运行。E23 是合同 smoke；GEARS 需要 gene-perturbation manifest。</p>
<figure><img src="../figures/model_family_compatibility_audit.svg" alt="model family compatibility audit"></figure>
<section class="card"><h2>E23 compatibility</h2>{html_table(e23_summary)}</section>
<section class="card"><h2>Perturbation class</h2>{html_table(class_counts)}</section>
<section class="card"><h2>GEARS candidate summary</h2>{html_table(gears_summary)}</section>
<section class="card"><h2>Environment</h2>{html_table(env)}</section>
<section class="card"><h2>Decisions</h2>{html_table(decisions)}</section>
</main></body></html>
"""
    (REPORTS / "E24_MODEL_FAMILY_COMPATIBILITY_AUDIT.html").write_text(html_doc, encoding="utf-8")


def write_readme(e23_summary: pd.DataFrame, decisions: pd.DataFrame) -> None:
    text = f"""# E24 model-family compatibility audit

先看结论：

- E23 manifest 有 {int(e23_summary['n_task_groups'].iloc[0])} 个 task groups，但 perturbation 是 Hpoly/Salmonella 这类 stimulus/timecourse。
- 因此 E23 适合作为 PredictionRecord 合同 smoke，不适合直接作为 GEARS biological benchmark。
- `scgpt_env` 可以 import GEARS；本地有 Norman/Adamson/Dixit/Frangieh processed GEARS assets。
- 下一步应做 GEARS gene-perturbation strict smoke，而不是 GEARS-on-E23。

入口：

- HTML 报告：`reports/E24_MODEL_FAMILY_COMPATIBILITY_AUDIT.html`
- Markdown 报告：`reports/E24_MODEL_FAMILY_COMPATIBILITY_AUDIT_REPORT.md`
- GEARS candidate manifest：`tables/GEARS_COMPATIBLE_CANDIDATE_TASKS.csv`
"""
    (OUT / "README_先看这个.md").write_text(text, encoding="utf-8")


def main() -> None:
    for path in (TABLES, REPORTS, FIGURES):
        path.mkdir(parents=True, exist_ok=True)
    e23_summary, class_counts = audit_e23_manifest()
    gears_records = collect_gears_candidate_records()
    gears_summary, candidate_manifest = build_gears_candidate_summary(gears_records)
    h5ad_inventory = inspect_processed_h5ad_inventory()
    env = build_environment_table()
    decisions = build_decision_table()

    e23_summary.to_csv(TABLES / "E23_MANIFEST_COMPATIBILITY_SUMMARY.csv", index=False)
    class_counts.to_csv(TABLES / "E23_PERTURBATION_CLASS_COUNTS.csv", index=False)
    gears_summary.to_csv(TABLES / "GEARS_COMPATIBLE_LEGACY_SUMMARY.csv", index=False)
    candidate_manifest.to_csv(TABLES / "GEARS_COMPATIBLE_CANDIDATE_TASKS.csv", index=False)
    h5ad_inventory.to_csv(TABLES / "GEARS_PROCESSED_H5AD_INVENTORY.csv", index=False)
    env.to_csv(TABLES / "MODEL_RUNTIME_ENVIRONMENT_CHECK.csv", index=False)
    decisions.to_csv(TABLES / "MODEL_FAMILY_DECISION_TABLE.csv", index=False)

    write_svg(e23_summary, env)
    write_reports(e23_summary, class_counts, gears_summary, candidate_manifest, h5ad_inventory, env, decisions)
    write_readme(e23_summary, decisions)
    status = {
        "status": "ok",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "git_head": git_head(),
        "git_dirty": git_dirty(),
        "e23_gears_compatible": bool(e23_summary["gears_compatible"].iloc[0]),
        "scgpt_env_gears_importable": bool(env.loc[env["environment"].eq("scgpt_env"), "gears_importable"].iloc[0]),
        "n_gears_candidate_rows": int(len(candidate_manifest)),
        "out": rel(OUT),
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
