#!/usr/bin/env python3
"""E23 package: shared benchmark adapter workbench.

This package freezes the small strict-pass E22 generator smoke as an adapter
development benchmark. It is not a new biology result. It defines what GEARS,
scGPT, CPA/chemCPA adapters must consume and emit before model-specific
reliability can be claimed.
"""

from __future__ import annotations

import html
import json
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "runtime" / "e22_generator_strict_smoke_20260707"
OUT = ROOT / "docs" / "实验结果" / "E23_shared_benchmark_adapter_workbench_20260708"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"
FIGURES = OUT / "figures"
INPUT = OUT / "input"

SCGPT_ENV = Path("/home/yyf/.conda/envs/scgpt_env")
SCGPT_ZIP = Path("/home/yyf/archive/code/20260519_0958_home_cleanup/moved_top_level/scGPT-main.zip")
CPA_PDF = Path("/home/yyf/safeconf_runtime/outputs/current_project_explanation_20260530/合并版_给我看的/papers/CPA_2023_MSB.pdf")
GEARS_FORMAL = Path("/home/yyf/safeconf_runtime/outputs/gears_prediction_records_formal")

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
        if str(path).startswith("/home/yyf/safeconf_runtime"):
            return "$SAFECONF_RUNTIME" + str(path)[len("/home/yyf/safeconf_runtime") :]
        if str(path).startswith("/home/yyf/.conda"):
            return "$YYF_CONDA" + str(path)[len("/home/yyf/.conda") :]
        if str(path).startswith("/home/yyf/archive"):
            return "$YYF_ARCHIVE" + str(path)[len("/home/yyf/archive") :]
        return str(path)


def run_python(env_python: Path, code: str) -> str:
    if not env_python.exists():
        return "python_missing"
    proc = subprocess.run(
        [str(env_python), "-c", code],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=20,
    )
    return proc.stdout.strip()


def inspect_scgpt_zip() -> dict[str, Any]:
    out: dict[str, Any] = {
        "exists": SCGPT_ZIP.exists(),
        "n_files": 0,
        "has_package_dir": False,
        "has_perturbation_tutorial": False,
        "has_pyproject": False,
        "zip_root": "",
    }
    if not SCGPT_ZIP.exists():
        return out
    with zipfile.ZipFile(SCGPT_ZIP) as zf:
        names = zf.namelist()
    out["n_files"] = len(names)
    out["has_package_dir"] = any(name.startswith("scGPT-main/scgpt/") for name in names)
    out["has_perturbation_tutorial"] = any("Tutorial_Perturbation" in name for name in names)
    out["has_pyproject"] = "scGPT-main/pyproject.toml" in names
    out["zip_root"] = "scGPT-main"
    return out


def build_task_manifest(records: pd.DataFrame) -> pd.DataFrame:
    manifest = (
        records.sort_values(GROUP_COLS + ["predictor_name"])
        .groupby(GROUP_COLS, dropna=False)
        .agg(
            task_id=("task_id", "first"),
            dataset_group=("dataset_group", "first"),
            gene_panel_id=("gene_panel_id", "first"),
            gene_order_hash=("gene_order_hash", "first"),
            effect_definition=("effect_definition", "first"),
            normalization_id=("normalization_id", "first"),
            error_normalization=("error_normalization", "first"),
            true_effect_key=("true_effect_key", "first"),
            target_control_key=("target_control_key", "first"),
            n_reference_predictors=("predictor_name", "nunique"),
            reference_predictors=("predictor_name", lambda x: ",".join(sorted(map(str, x.unique())))),
        )
        .reset_index()
    )
    manifest.insert(0, "benchmark_id", "E23_HABER_200GENE_STRICT_SMOKE")
    manifest["source_prediction_records"] = rel(RUNTIME / "tables" / "PREDICTION_RECORDS.csv")
    manifest["source_true_npz"] = rel(RUNTIME / "input" / "true_effects.npz")
    manifest["source_control_npz"] = rel(RUNTIME / "input" / "target_control_means.npz")
    manifest["required_adapter_output"] = "PREDICTION_RECORDS.csv + predicted_effects.npz; reuse task-scoped true_effect_key"
    return manifest


def validate_manifest(manifest: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.append(
        {
            "check": "one_true_key_per_task_group",
            "status": "pass" if manifest["true_effect_key"].nunique() == len(manifest) else "fail",
            "value": f"{manifest['true_effect_key'].nunique()}/{len(manifest)}",
            "why_it_matters": "不同模型必须共享同一个 true effect，才能比较模型特异错误。",
        }
    )
    rows.append(
        {
            "check": "records_are_two_predictor_reference",
            "status": "pass" if records["predictor_name"].nunique() >= 2 else "fail",
            "value": ",".join(sorted(records["predictor_name"].astype(str).unique())),
            "why_it_matters": "adapter 开发时保留参考预测器，便于 sanity check。",
        }
    )
    rows.append(
        {
            "check": "single_gene_space",
            "status": "pass" if manifest["gene_order_hash"].nunique() == 1 else "fail",
            "value": str(manifest["gene_order_hash"].nunique()),
            "why_it_matters": "跨模型向量必须处在同一 gene order 上。",
        }
    )
    rows.append(
        {
            "check": "single_effect_definition",
            "status": "pass" if manifest["effect_definition"].nunique() == 1 else "fail",
            "value": ",".join(sorted(manifest["effect_definition"].astype(str).unique())),
            "why_it_matters": "mean_diff、logFC 等 effect 定义不能混用。",
        }
    )
    rows.append(
        {
            "check": "contains_test_split",
            "status": "pass" if "test" in set(manifest["split"].astype(str)) else "fail",
            "value": ",".join(sorted(manifest["split"].astype(str).unique())),
            "why_it_matters": "adapter smoke 至少要覆盖 test 行。",
        }
    )
    return pd.DataFrame(rows)


def build_asset_inventory() -> pd.DataFrame:
    scgpt_import = run_python(
        SCGPT_ENV / "bin" / "python",
        "import importlib.util; print(bool(importlib.util.find_spec('scgpt')))",
    )
    scgpt_zip = inspect_scgpt_zip()
    return pd.DataFrame(
        [
            {
                "asset": "GEARS legacy formal vectors",
                "kind": "prediction_records_and_npz",
                "path": rel(GEARS_FORMAL),
                "exists": GEARS_FORMAL.exists(),
                "usable_now": True,
                "limitation": "旧 GEARS 输出是 legacy/non-strict；需要按 shared manifest 重跑或转换。",
            },
            {
                "asset": "Patched GEARS exporter",
                "kind": "code",
                "path": "code/20260426_154505_perturb_transport_final_push/safetrans_confidence/cli/run_gears_prediction_records.py",
                "exists": (ROOT / "code/20260426_154505_perturb_transport_final_push/safetrans_confidence/cli/run_gears_prediction_records.py").exists(),
                "usable_now": True,
                "limitation": "能写 strict provenance；尚未对 E23 shared manifest 运行。",
            },
            {
                "asset": "scGPT conda env",
                "kind": "environment",
                "path": rel(SCGPT_ENV),
                "exists": SCGPT_ENV.exists(),
                "usable_now": scgpt_import.strip().lower() == "true",
                "limitation": f"scgpt import result: {scgpt_import}; env 有 torch/scanpy 但未安装 scgpt 包。",
            },
            {
                "asset": "scGPT source archive",
                "kind": "source_zip",
                "path": rel(SCGPT_ZIP),
                "exists": bool(scgpt_zip["exists"]),
                "usable_now": bool(scgpt_zip["exists"] and scgpt_zip["has_package_dir"]),
                "limitation": f"zip files={scgpt_zip['n_files']}; has perturbation tutorial={scgpt_zip['has_perturbation_tutorial']}; no local checkpoint found.",
            },
            {
                "asset": "CPA / chemCPA local executable output",
                "kind": "missing_adapter_asset",
                "path": rel(CPA_PDF),
                "exists": CPA_PDF.exists(),
                "usable_now": False,
                "limitation": "本地只有论文 PDF 痕迹，没有可运行代码、checkpoint 或 SafeConf PredictionRecord。",
            },
        ]
    )


def build_model_backlog() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "SafeConf reference baselines",
                "current_state": "E22 strict-pass generator output available",
                "next_adapter_step": "Use E23 manifest as the shared reference input.",
                "claim_allowed_now": "Can be used as adapter smoke and sanity baseline.",
                "claim_not_allowed": "Not a deep model comparison.",
                "priority": 1,
            },
            {
                "model": "GEARS",
                "current_state": "Legacy vectors exist; exporter patched for provenance",
                "next_adapter_step": "Add a GEARS runner that reads SHARED_BENCHMARK_TASK_MANIFEST.csv and writes strict PredictionRecord on the same gene order.",
                "claim_allowed_now": "GEARS-only supplementary legacy evidence exists.",
                "claim_not_allowed": "Unified GEARS/scGPT/CPA validation.",
                "priority": 2,
            },
            {
                "model": "scGPT",
                "current_state": "Source zip exists; env exists; package import currently false",
                "next_adapter_step": "Unpack/install source or point PYTHONPATH to source; locate perturbation checkpoint/tutorial assets; export predicted_effects on E23 manifest.",
                "claim_allowed_now": "Local source availability only.",
                "claim_not_allowed": "scGPT prediction-vector validation.",
                "priority": 3,
            },
            {
                "model": "CPA / chemCPA",
                "current_state": "Only literature PDF trace found locally",
                "next_adapter_step": "Acquire executable implementation/checkpoint or replace with another accessible perturbation model for shared benchmark.",
                "claim_allowed_now": "Future adapter target.",
                "claim_not_allowed": "CPA validation or comparison.",
                "priority": 4,
            },
        ]
    )


def build_output_schema() -> pd.DataFrame:
    rows = [
        ("record_id", "string", "adapter-specific unique prediction row id"),
        ("task_id", "int/string", "reuse task id from manifest when possible"),
        ("task_key", "string", "must match manifest"),
        ("dataset_name", "string", "must match manifest"),
        ("fold_id", "int", "must match manifest"),
        ("split", "train/val/test", "must match manifest"),
        ("context", "string", "must match manifest"),
        ("perturbation", "string", "must match manifest"),
        ("predictor_name", "string", "GEARS/scGPT/CPA/etc."),
        ("gene_panel_id", "string", "must match manifest"),
        ("gene_order_hash", "sha256", "must match manifest exactly"),
        ("effect_definition", "string", "must match manifest"),
        ("normalization_id", "string", "must match manifest"),
        ("predicted_effect_key", "string", "key in predicted_effects.npz"),
        ("true_effect_key", "string", "reuse task-scoped key from manifest"),
        ("true_error_rmse", "float", "computed after predicted vector is exported"),
    ]
    return pd.DataFrame(rows, columns=["field", "type", "requirement"])


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


def write_svg(manifest: pd.DataFrame, asset_inventory: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    ready = int(asset_inventory["usable_now"].sum())
    total = len(asset_inventory)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1020" height="330" viewBox="0 0 1020 330">
<rect width="1020" height="330" fill="#fbfbf8"/>
<text x="42" y="38" font-size="23" font-weight="800" fill="#17212b">E23 shared benchmark adapter workbench</text>
<text x="42" y="62" font-size="14" fill="#5d6974">A strict-pass task manifest becomes the contract for GEARS, scGPT, CPA/chemCPA adapters.</text>
<g transform="translate(42 98)">
<rect x="0" y="0" width="210" height="64" rx="8" fill="#fff" stroke="#d8e0e4"/>
<text x="105" y="25" text-anchor="middle" font-size="15" font-weight="700">Task manifest</text>
<text x="105" y="47" text-anchor="middle" font-size="12" fill="#5d6974">{len(manifest)} task groups · strict source</text>
<text x="235" y="39" font-size="24" fill="#087f73">→</text>
<rect x="275" y="0" width="190" height="64" rx="8" fill="#fff" stroke="#d8e0e4"/>
<text x="370" y="25" text-anchor="middle" font-size="15" font-weight="700">Model adapters</text>
<text x="370" y="47" text-anchor="middle" font-size="12" fill="#5d6974">GEARS / scGPT / CPA</text>
<text x="490" y="39" font-size="24" fill="#087f73">→</text>
<rect x="530" y="0" width="210" height="64" rx="8" fill="#fff" stroke="#d8e0e4"/>
<text x="635" y="25" text-anchor="middle" font-size="15" font-weight="700">PredictionRecord</text>
<text x="635" y="47" text-anchor="middle" font-size="12" fill="#5d6974">same true key · same gene order</text>
<text x="765" y="39" font-size="24" fill="#087f73">→</text>
<rect x="805" y="0" width="145" height="64" rx="8" fill="#fff" stroke="#d8e0e4"/>
<text x="877" y="25" text-anchor="middle" font-size="15" font-weight="700">SafeConf</text>
<text x="877" y="47" text-anchor="middle" font-size="12" fill="#5d6974">model-specific audit</text>
</g>
<g transform="translate(42 210)">
<rect x="0" y="0" width="360" height="48" rx="8" fill="#e7f3eb" stroke="#c8dfd0"/>
<text x="20" y="30" font-size="15" fill="#28734f" font-weight="700">Usable local assets: {ready}/{total}</text>
<rect x="420" y="0" width="470" height="48" rx="8" fill="#fff3d6" stroke="#ead89a"/>
<text x="440" y="30" font-size="15" fill="#946205" font-weight="700">Current blocker: scGPT/CPA do not yet emit strict PredictionRecords.</text>
</g>
</svg>
"""
    (FIGURES / "shared_benchmark_adapter_workbench.svg").write_text(svg, encoding="utf-8")


def write_reports(
    manifest: pd.DataFrame,
    checks: pd.DataFrame,
    assets: pd.DataFrame,
    backlog: pd.DataFrame,
    schema: pd.DataFrame,
) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    report = f"""# E23 shared benchmark adapter workbench

生成时间：{now}

## 1. 目的

E23 固化一个小型 strict-pass shared benchmark。以后 GEARS、scGPT、CPA/chemCPA 不再各跑各的，而是必须对 `SHARED_BENCHMARK_TASK_MANIFEST.csv` 逐任务输出预测向量。

## 2. Manifest checks

{md_table(checks)}

## 3. Local asset inventory

{md_table(assets)}

## 4. Adapter backlog

{md_table(backlog)}

## 5. Required output schema

{md_table(schema)}

## 6. 结论

- E23 已给出 120 个 task groups 的 shared benchmark manifest。
- GEARS 有 legacy 结果和 patched exporter，是最先值得接到 manifest 的模型。
- scGPT 有源码压缩包和 conda 环境，但当前 `import scgpt` 为 false，且没有本地 checkpoint / PredictionRecord。
- CPA/chemCPA 当前只有论文 PDF 痕迹，没有本地可执行向量输出。
- 因此下一项实质工作应是 GEARS-on-E23 或 scGPT source install + adapter smoke，不能写成三模型统一验证已经完成。
"""
    (REPORTS / "E23_SHARED_BENCHMARK_ADAPTER_WORKBENCH_REPORT.md").write_text(report, encoding="utf-8")
    html_doc = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>E23 shared benchmark adapter workbench</title>
<style>
body{{margin:0;background:#fbfbf8;color:#17212b;font-family:"Microsoft YaHei","PingFang SC","Segoe UI",Arial,sans-serif;line-height:1.72}}
.wrap{{max-width:1180px;margin:0 auto;padding:34px 28px 70px}}h1{{margin:0 0 8px;font-size:34px}}h2{{margin-top:30px;border-bottom:2px solid #d8e0e4;padding-bottom:8px}}
.lead{{color:#5d6974;max-width:940px}}.card,figure{{background:#fff;border:1px solid #d8e0e4;border-radius:10px;padding:18px;margin:16px 0;overflow-x:auto;box-shadow:0 8px 24px rgba(28,42,52,.06)}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #d8e0e4;padding:7px 8px;vertical-align:top}}th{{background:#eaf0f1;text-align:left}}img{{max-width:100%}}code{{background:#f4f7f7;border:1px solid #dce4e7;border-radius:4px;padding:2px 5px}}
</style></head><body><main class="wrap">
<h1>E23 shared benchmark adapter workbench</h1>
<p class="lead">这里把 model-specific validation 的第一块地基固定下来：所有模型必须对同一个 task manifest 输出同格式 PredictionRecord。当前这是工程工作台，不是三模型结果。</p>
<figure><img src="../figures/shared_benchmark_adapter_workbench.svg" alt="shared benchmark adapter workbench"></figure>
<section class="card"><h2>Manifest checks</h2>{html_table(checks)}</section>
<section class="card"><h2>Local assets</h2>{html_table(assets)}</section>
<section class="card"><h2>Adapter backlog</h2>{html_table(backlog)}</section>
<section class="card"><h2>Output schema</h2>{html_table(schema)}</section>
<section class="card"><h2>Task manifest sample</h2>{html_table(manifest, n=20)}</section>
</main></body></html>
"""
    (REPORTS / "E23_SHARED_BENCHMARK_ADAPTER_WORKBENCH.html").write_text(html_doc, encoding="utf-8")


def write_readme(manifest: pd.DataFrame, checks: pd.DataFrame) -> None:
    text = f"""# E23 shared benchmark adapter workbench

先看结论：

- E23 不是新的模型结果，而是 model-specific validation 的 shared benchmark 地基。
- 已生成 `input/SHARED_BENCHMARK_TASK_MANIFEST.csv`，共 {len(manifest)} 个 task groups。
- manifest 检查：{int(checks['status'].eq('pass').sum())}/{len(checks)} pass。
- 现在可以开始写 GEARS/scGPT/CPA adapter，但不能说三模型统一验证已经完成。

入口：

- HTML 报告：`reports/E23_SHARED_BENCHMARK_ADAPTER_WORKBENCH.html`
- Markdown 报告：`reports/E23_SHARED_BENCHMARK_ADAPTER_WORKBENCH_REPORT.md`
- 任务 manifest：`input/SHARED_BENCHMARK_TASK_MANIFEST.csv`
- 输出 schema：`tables/ADAPTER_REQUIRED_OUTPUT_SCHEMA.csv`
"""
    (OUT / "README_先看这个.md").write_text(text, encoding="utf-8")


def main() -> None:
    for path in (TABLES, REPORTS, FIGURES, INPUT):
        path.mkdir(parents=True, exist_ok=True)
    records = pd.read_csv(RUNTIME / "tables" / "PREDICTION_RECORDS.csv")
    manifest = build_task_manifest(records)
    checks = validate_manifest(manifest, records)
    assets = build_asset_inventory()
    backlog = build_model_backlog()
    schema = build_output_schema()
    manifest.to_csv(INPUT / "SHARED_BENCHMARK_TASK_MANIFEST.csv", index=False)
    checks.to_csv(TABLES / "SHARED_BENCHMARK_MANIFEST_CHECKS.csv", index=False)
    assets.to_csv(TABLES / "LOCAL_MODEL_ASSET_INVENTORY.csv", index=False)
    backlog.to_csv(TABLES / "MODEL_ADAPTER_BACKLOG.csv", index=False)
    schema.to_csv(TABLES / "ADAPTER_REQUIRED_OUTPUT_SCHEMA.csv", index=False)
    write_svg(manifest, assets)
    write_reports(manifest, checks, assets, backlog, schema)
    write_readme(manifest, checks)
    status = {
        "status": "ok" if checks["status"].eq("pass").all() else "failed",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "git_head": git_head(),
        "git_dirty": git_dirty(),
        "n_task_groups": int(len(manifest)),
        "manifest_checks_pass": int(checks["status"].eq("pass").sum()),
        "manifest_checks_total": int(len(checks)),
        "out": rel(OUT),
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
