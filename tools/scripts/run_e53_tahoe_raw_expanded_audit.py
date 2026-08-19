#!/usr/bin/env python3
"""E53 expanded Tahoe raw stratification audit.

The E41 audit proved that Tahoe raw shards contain the fields we need.  E53
checks whether the currently downloaded raw layer has enough coverage for the
advisor-requested multidimensional follow-up:

* drug holdout
* cell-line holdout
* MoA holdout
* plate/batch audit
* drug x cell-line and MoA x cell-line feasibility

It reads only metadata columns from a balanced sample of complete shards.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
ATLAS = Path("/home/yyf/data/singlecell_perturbation_atlas")
TAHOE_RAW = ATLAS / "mega_external" / "Tahoe-100M" / "data"
TAHOE_META = ATLAS / "mega_external" / "Tahoe-100M" / "metadata"
OUT = ROOT / "docs" / "实验结果" / "E53_tahoe_raw_expanded_audit_20260710"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"

FIELDS = ["drug", "cell_line_id", "moa-fine", "canonical_smiles", "pubchem_cid", "plate"]


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        return bool(subprocess.check_output(["git", "status", "--short"], cwd=ROOT).decode().strip())
    except Exception:
        return True


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except Exception:
        try:
            return path.relative_to(ATLAS).as_posix()
        except Exception:
            return str(path)


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)


def complete_shards() -> list[Path]:
    shards = sorted(TAHOE_RAW.glob("train-*.parquet"))
    return [p for p in shards if not Path(str(p) + ".aria2").exists()]


def balanced_sample(items: list[Path], n: int) -> list[Path]:
    if len(items) <= n:
        return items
    idx = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[i] for i in sorted(set(idx.tolist()))]


def read_sample(shards: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    shard_rows = []
    for path in shards:
        pf = pq.ParquetFile(path)
        cols = [c for c in FIELDS if c in pf.schema_arrow.names]
        table = pq.read_table(path, columns=cols)
        df = table.to_pandas()
        df["_source_file"] = path.name
        frames.append(df)
        shard_rows.append(
            {
                "file_name": path.name,
                "relative_path": rel(path),
                "n_rows": int(len(df)),
                "columns_read": ";".join(cols),
            }
        )
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return data, pd.DataFrame(shard_rows)


def field_coverage(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for field in FIELDS:
        if field not in data:
            rows.append({"field": field, "status": "missing"})
            continue
        s = data[field]
        vc = s.astype(str).replace({"": "EMPTY"}).value_counts(dropna=False)
        rows.append(
            {
                "field": field,
                "status": "ok",
                "n_rows": int(len(s)),
                "n_missing_or_empty": int(s.isna().sum() + (s.astype(str).str.len() == 0).sum()),
                "n_unique": int(s.nunique(dropna=True)),
                "top20": " | ".join(f"{k}:{int(v)}" for k, v in vc.head(20).items()),
            }
        )
    return pd.DataFrame(rows)


def combo_feasibility(data: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("drug", ["drug"]),
        ("cell_line", ["cell_line_id"]),
        ("moa", ["moa-fine"]),
        ("plate", ["plate"]),
        ("drug_x_cell_line", ["drug", "cell_line_id"]),
        ("moa_x_cell_line", ["moa-fine", "cell_line_id"]),
        ("drug_x_plate", ["drug", "plate"]),
        ("cell_line_x_plate", ["cell_line_id", "plate"]),
    ]
    rows = []
    for name, cols in specs:
        if any(c not in data for c in cols):
            continue
        sub = data[cols].copy()
        for c in cols:
            sub[c] = sub[c].astype(str).replace({"": "EMPTY"})
        counts = sub.groupby(cols, dropna=False, observed=False).size().reset_index(name="n_cells")
        rows.append(
            {
                "dimension": name,
                "columns": "+".join(cols),
                "n_groups": int(len(counts)),
                "n_groups_ge_50": int((counts["n_cells"] >= 50).sum()),
                "n_groups_ge_100": int((counts["n_cells"] >= 100).sum()),
                "n_groups_ge_300": int((counts["n_cells"] >= 300).sum()),
                "n_groups_ge_900": int((counts["n_cells"] >= 900).sum()),
                "median_cells_per_group": float(counts["n_cells"].median()),
                "max_cells_per_group": int(counts["n_cells"].max()),
                "top10_groups": " | ".join(
                    [
                        f"{' / '.join(map(str, row[cols].tolist()))}:{int(row['n_cells'])}"
                        for _, row in counts.sort_values("n_cells", ascending=False).head(10).iterrows()
                    ]
                ),
            }
        )
    return pd.DataFrame(rows)


def metadata_status() -> pd.DataFrame:
    rows = []
    for name in ["gene_metadata.parquet", "obs_metadata.parquet"]:
        path = TAHOE_META / name
        if not path.exists():
            rows.append({"file_name": name, "status": "missing"})
            continue
        pf = pq.ParquetFile(path)
        rows.append(
            {
                "file_name": name,
                "status": "ok",
                "n_rows": int(pf.metadata.num_rows),
                "columns": ";".join(pf.schema_arrow.names),
                "relative_path": rel(path),
            }
        )
    return pd.DataFrame(rows)


def write_report(status: dict, coverage: pd.DataFrame, combo: pd.DataFrame) -> None:
    lines = []
    lines.append("# E53 Tahoe raw 扩展字段审计\n")
    lines.append(f"- 生成时间：{now_text()}")
    lines.append(f"- Git：`{git_head()[:12]}`")
    lines.append(f"- 工作区 dirty：`{git_dirty()}`")
    lines.append(f"- 完整 shard：{status['complete_shards_seen']}；本次审计 shard：{status['audited_shards']}；审计行数：{status['audited_rows']}\n")
    lines.append("## 字段覆盖\n")
    lines.append(coverage.to_string(index=False))
    lines.append("\n## 组合可行性\n")
    lines.append(combo.to_string(index=False))
    lines.append("\n## 怎么用\n")
    lines.append("- 如果 `drug_x_cell_line` 和 `moa_x_cell_line` 有足够多 group，就可以接 Tahoe raw 的 cell-line / drug / MoA 分层。")
    lines.append("- `plate` 能作为 batch 线索，用来检查高风险是否只是 plate 驱动。")
    lines.append("- raw 仍在下载，这份结果是当前完整 shard 的快照，不替代全量 Tahoe。")
    (REPORTS / "E53_TAHOE_RAW_EXPANDED_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "README_先看这个.md").write_text(
        "# E53 Tahoe raw 扩展字段审计\n\n"
        "先看 `reports/E53_TAHOE_RAW_EXPANDED_AUDIT.md`。\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-shards", type=int, default=128)
    args = parser.parse_args()
    ensure_dirs()
    complete = complete_shards()
    selected = balanced_sample(complete, args.max_shards)
    data, shard_df = read_sample(selected)
    coverage = field_coverage(data)
    combo = combo_feasibility(data)
    meta = metadata_status()
    shard_df.to_csv(TABLES / "TAHOE_RAW_AUDITED_SHARDS.csv", index=False)
    coverage.to_csv(TABLES / "TAHOE_RAW_FIELD_COVERAGE.csv", index=False)
    combo.to_csv(TABLES / "TAHOE_RAW_COMBO_FEASIBILITY.csv", index=False)
    meta.to_csv(TABLES / "TAHOE_METADATA_STATUS.csv", index=False)
    status = {
        "generated_at": now_text(),
        "git_head": git_head(),
        "git_dirty": git_dirty(),
        "complete_shards_seen": int(len(complete)),
        "partial_shard_markers": int(len(list(TAHOE_RAW.glob("*.aria2")))),
        "audited_shards": int(len(selected)),
        "audited_rows": int(len(data)),
        "output_dir": rel(OUT),
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(status, coverage, combo)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
