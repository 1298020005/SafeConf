#!/usr/bin/env python3
"""Build an E40 inventory for non-Tahoe and newly acquired public datasets.

The goal is not to run a model.  It gives the project a clean ledger of the
data dimensions now available for advisor-requested follow-up experiments:
genetic, chemical, combinatorial, regulatory/enhancer, cross-context, and
large-scale external panels.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


ATLAS = Path("/home/yyf/data/singlecell_perturbation_atlas")
PROJECT = Path("/home/yyf/proj")
OUT = PROJECT / "docs/实验结果/E40_non_tahoe_multidim_data_acquisition_20260709"
TABLES = OUT / "tables"
SCAN = ATLAS / "metadata/h5ad_scan.tsv"
TAHOE_DE = ATLAS / "mega_external/Tahoe-100M/metadata/pseudobulk_differential_expression"
TAHOE_STATUS = ATLAS / "mega_external/Tahoe-100M/download_logs/TAHOE_RAW_DOWNLOAD_STATUS.json"
OPENPROBLEMS_STATUS = (
    ATLAS
    / "mega_external/OpenProblems_NeurIPS2023_single_cell_perturbations"
    / "download_logs/OPENPROBLEMS_NEURIPS2023_DOWNLOAD_STATUS.json"
)

BUCKET = {
    "genetic_single": "基因单扰动",
    "genetic_combinatorial": "基因组合扰动",
    "chemical_single": "药物/化学单扰动",
    "chemical_combinatorial": "药物/化学组合扰动",
    "enhancer_regulatory": "增强子/调控元件扰动",
}


def safe_int_sum(s: pd.Series) -> int:
    return int(pd.to_numeric(s, errors="coerce").fillna(0).sum())


def read_json(path: Path) -> dict:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["exists"] = True
        data["path"] = str(path)
        return data
    except Exception as exc:
        return {"exists": True, "path": str(path), "read_error": repr(exc)}


def build_local_inventory(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["source_name", "study_family", "perturbation_type"]
    for keys, g in df.groupby(group_cols, dropna=False):
        source_name, study_family, ptype = keys
        examples = []
        for p in g["local_path"].head(3):
            try:
                examples.append(str(Path(p).relative_to(ATLAS)))
            except Exception:
                examples.append(str(p))
        rows.append(
            {
                "source_name": source_name,
                "study_family": study_family,
                "perturbation_type": ptype,
                "dimension_bucket_cn": BUCKET.get(str(ptype), str(ptype)),
                "n_files": len(g),
                "total_cells_or_obs": safe_int_sum(g["n_obs"]),
                "median_genes_or_vars": int(pd.to_numeric(g["n_vars"], errors="coerce").median()),
                "total_file_size_gb": round(safe_int_sum(g["file_size_bytes"]) / 1e9, 3),
                "has_cell_type_files": int(g["has_cell_type"].astype(str).str.lower().eq("true").sum()),
                "has_donor_files": int(g["has_donor"].astype(str).str.lower().eq("true").sum()),
                "example_relative_paths": " | ".join(examples),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["dimension_bucket_cn", "source_name", "study_family"], kind="stable"
    )


def build_source_coverage(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ptype, g in df.groupby("perturbation_type", dropna=False):
        rows.append(
            {
                "perturbation_type": ptype,
                "dimension_bucket_cn": BUCKET.get(str(ptype), str(ptype)),
                "n_files": len(g),
                "n_study_families": g["study_family"].nunique(dropna=True),
                "total_cells_or_obs": safe_int_sum(g["n_obs"]),
                "total_file_size_gb": round(safe_int_sum(g["file_size_bytes"]) / 1e9, 3),
                "representative_studies": ", ".join(
                    list(g["study_family"].dropna().astype(str).value_counts().head(8).index)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("n_files", ascending=False, kind="stable")


def external_status_rows() -> pd.DataFrame:
    rows = []

    tahoe_de_files = list(TAHOE_DE.glob("*.parquet")) if TAHOE_DE.exists() else []
    rows.append(
        {
            "external_dataset": "Tahoe-100M pseudobulk differential expression",
            "dimension": "大规模化学扰动 / pseudobulk DE",
            "local_status": f"{len(tahoe_de_files)}/1026 parquet present",
            "downloaded_size_gb": round(sum(p.stat().st_size for p in tahoe_de_files) / 1e9, 3),
            "local_relative_path": str(TAHOE_DE.relative_to(ATLAS)) if TAHOE_DE.exists() else "",
            "role_for_safeconf": "已用于 Tahoe chemical D1-D5；可继续做细胞系/药物/分片稳定性。",
        }
    )

    tahoe = read_json(TAHOE_STATUS)
    rows.append(
        {
            "external_dataset": "Tahoe-100M raw single-cell shards",
            "dimension": "大规模单细胞原始层 / raw parquet shards",
            "local_status": (
                f"{tahoe.get('complete_parquet_files', 0)} complete, "
                f"{tahoe.get('partial_parquet_files', 0)} partial, "
                f"{tahoe.get('expected_shards', 3388)} expected"
            ),
            "downloaded_size_gb": tahoe.get("downloaded_size_gb", 0),
            "local_relative_path": "mega_external/Tahoe-100M/data",
            "role_for_safeconf": "正在补原始层，后续可做更细粒度细胞状态、细胞系和药物上下文检查。",
        }
    )

    op = read_json(OPENPROBLEMS_STATUS)
    if "files" in op:
        local_status = f"{op.get('n_complete', 0)}/{op.get('n_files', 6)} complete, {op.get('n_partial', 0)} partial"
        downloaded = op.get("disk_usage_gb", op.get("downloaded_size_gb", 0))
    else:
        local_status = "not checked"
        downloaded = 0
    rows.append(
        {
            "external_dataset": "OpenProblems / NeurIPS 2023 single-cell perturbations",
            "dimension": "PBMC 小分子扰动 / 多供体 / 多细胞类型 / baseline multiome 背景",
            "local_status": local_status,
            "downloaded_size_gb": downloaded,
            "local_relative_path": "mega_external/OpenProblems_NeurIPS2023_single_cell_perturbations",
            "role_for_safeconf": "新补数据源，可做独立化学扰动、跨细胞类型、跨供体、比赛式 held-out 条件。",
        }
    )
    return pd.DataFrame(rows)


def write_readme(local_inv: pd.DataFrame, coverage: pd.DataFrame, external: pd.DataFrame) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    lines = []
    lines.append("# E40 非 Tahoe 多维数据补强记录\n")
    lines.append(f"- 生成时间：{now}")
    lines.append("- 目的：把老师要求的“更多数据、多维度数据”先落到可追踪的数据账本里。")
    lines.append("- 这份记录不替代实验结果，它回答：现在手里有哪些数据，哪些已经完整，哪些正在下载，下一步能设计哪些实验。\n")

    lines.append("## 1. 当前数据层次\n")
    for _, row in coverage.iterrows():
        lines.append(
            f"- {row['dimension_bucket_cn']}：{int(row['n_files'])} 个 h5ad，"
            f"{int(row['n_study_families'])} 个 study family，"
            f"约 {int(row['total_cells_or_obs']):,} 个观测；代表数据：{row['representative_studies']}"
        )

    lines.append("\n## 2. 新补的外部数据\n")
    for _, row in external.iterrows():
        lines.append(
            f"- {row['external_dataset']}：{row['local_status']}，"
            f"已落盘约 {row['downloaded_size_gb']} GB；用途：{row['role_for_safeconf']}"
        )

    lines.append("\n## 3. 对后续实验的直接意义\n")
    lines.extend(
        [
            "- 基因单扰动：继续做 GEARS / scGPT / 简单基线的普通基因敲除风险评估。",
            "- 基因组合扰动：检查 SafeConf 面对组合扰动时，support、context、model disagreement 是否还能解释失败。",
            "- 化学扰动：Tahoe 和 OpenProblems 形成两个独立来源，可测试 chemical setting 是否只是 Tahoe 特例。",
            "- 增强子/调控元件扰动：用于补一个 regulatory/enhancer 方向，不让论文只停在 gene/drug 两类。",
            "- 多供体/多细胞类型：OpenProblems 可以直接设计 donor holdout、cell-type holdout、compound holdout。",
            "- 原始单细胞层：Tahoe raw 与 OpenProblems raw 后续能做更细的细胞状态分层，不只依赖 pseudobulk。\n",
        ]
    )

    lines.append("## 4. 文件说明\n")
    lines.extend(
        [
            "- `tables/non_tahoe_local_inventory.csv`：本地 scPerturb + scPerturBench 的 study-level 总表。",
            "- `tables/source_coverage_by_dimension.csv`：按扰动类型汇总的覆盖表。",
            "- `tables/external_acquisition_status.csv`：Tahoe raw、Tahoe pseudobulk、OpenProblems 的下载状态。",
            "- `RUN_STATUS.json`：机器可读状态记录。\n",
        ]
    )

    lines.append("## 5. 后续触发顺序\n")
    lines.extend(
        [
            "1. 等 OpenProblems 6 个文件完成后，先做 metadata scan，确认 `cell_type / donor / compound` 字段。",
            "2. 用 processed DGE 三件套做一个 OpenProblems chemical smoke：mean baseline、magnitude、SafeConf-style risk 三组先跑通。",
            "3. 若字段完整，再做 compound holdout、cell-type holdout、donor holdout 三个 split。",
            "4. Tahoe raw 不等全量完成也能先抽前若干 shard 做字段审计，确认原始层是否能支持更细分任务。\n",
        ]
    )
    (OUT / "README_先看这个.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    if not SCAN.exists():
        raise SystemExit(f"missing scan table: {SCAN}")
    df = pd.read_csv(SCAN, sep="\t")
    local_inv = build_local_inventory(df)
    coverage = build_source_coverage(df)
    external = external_status_rows()

    local_inv.to_csv(TABLES / "non_tahoe_local_inventory.csv", index=False)
    coverage.to_csv(TABLES / "source_coverage_by_dimension.csv", index=False)
    external.to_csv(TABLES / "external_acquisition_status.csv", index=False)
    write_readme(local_inv, coverage, external)

    status = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scan_table": str(SCAN),
        "local_h5ad_rows": len(df),
        "coverage_rows": len(coverage),
        "inventory_rows": len(local_inv),
        "external_rows": len(external),
        "output_dir": str(OUT.relative_to(PROJECT)),
        "tables": [
            "tables/non_tahoe_local_inventory.csv",
            "tables/source_coverage_by_dimension.csv",
            "tables/external_acquisition_status.csv",
        ],
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
