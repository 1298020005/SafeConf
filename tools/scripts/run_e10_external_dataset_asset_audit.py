#!/usr/bin/env python3
"""E10 external task-level validation asset audit.

This script consolidates external data resources for the next validation stage.
It does not run a heavy prediction model.  Its job is to make the data situation
unambiguous:

* what scPerturBench/scPerturb files are present on this server
* which official Zenodo files are covered
* which datasets are promising for task-level external validation
* what should be downloaded or run next
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/home/yyf/data/singlecell_perturbation_atlas")
LEGACY_INVENTORY = ROOT / "code/20260426_154505_perturb_transport_final_push/01_asset_audit/DATASET_INVENTORY.csv"
OUT = ROOT / "docs" / "实验结果" / "E10_external_task_validation_assets_20260707"
META_ROOT = OUT / "source_metadata"
SCPERTURBENCH_ZENODO = META_ROOT / "scperturbench_20260707/scperturbench_zenodo_14607156.json"
SCPERTURB_ZENODO = META_ROOT / "scperturb_20260707/scperturb_zenodo_13350497.json"
SCPERTURBENCH_README = META_ROOT / "scperturbench_20260707/scperturbench_readme.md"
SCPERTURBENCH_MD5_XLSX = META_ROOT / "scperturbench_20260707/scperturbench_md5sum.xlsx"

TABLES = OUT / "tables"
REPORTS = OUT / "reports"
DOWNLOADS = OUT / "download_manifests"

METADATA_DOWNLOADS = {
    SCPERTURBENCH_ZENODO: "https://zenodo.org/api/records/14607156",
    SCPERTURB_ZENODO: "https://zenodo.org/api/records/13350497",
    SCPERTURBENCH_README: "https://raw.githubusercontent.com/bm2-lab/scPerturBench/main/README.md",
    SCPERTURBENCH_MD5_XLSX: "https://raw.githubusercontent.com/bm2-lab/scPerturBench/main/scperturbench_v1_md5sum.xlsx",
}


def normalize_metadata_file(path: Path) -> None:
    if path.suffix.lower() != ".md":
        return
    lines = [line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()]
    while lines and lines[-1] == "":
        lines.pop()
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_metadata() -> None:
    """Materialize small upstream metadata files inside the docs artifact.

    Runtime caches are treated as a fallback only.  The committed docs artifact
    should carry the exact metadata snapshot used by the audit.
    """

    old_cache_root = ROOT / "runtime/external_metadata"
    for path, url in METADATA_DOWNLOADS.items():
        if not (path.exists() and path.stat().st_size > 0):
            path.parent.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                ["curl", "-L", "--fail", "--retry", "3", "-o", str(path), url],
                cwd=ROOT,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if not (result.returncode == 0 and path.exists() and path.stat().st_size > 0):
                cache_path = old_cache_root / path.relative_to(META_ROOT)
                if cache_path.exists() and cache_path.stat().st_size > 0:
                    shutil.copy2(cache_path, path)
                else:
                    raise RuntimeError(f"Failed to obtain metadata: {url}\n{result.stderr}")
        normalize_metadata_file(path)


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def load_zenodo(path: Path, source: str) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for f in data.get("files", []):
        links = f.get("links", {}) or {}
        rows.append(
            {
                "source": source,
                "record_title": data.get("metadata", {}).get("title", ""),
                "record_doi": data.get("doi", ""),
                "file_name": f.get("key", ""),
                "official_size_bytes": f.get("size"),
                "official_checksum": f.get("checksum", ""),
                "download_url": links.get("self") or links.get("content") or "",
            }
        )
    return pd.DataFrame(rows)


def scan_actual_files() -> pd.DataFrame:
    rows = []
    for p in sorted(DATA_ROOT.rglob("*.h5ad")):
        rel = p.relative_to(DATA_ROOT).as_posix()
        parts = p.relative_to(DATA_ROOT).parts
        group = parts[0] if parts else ""
        rows.append(
            {
                "actual_path": str(p),
                "relative_path": rel,
                "actual_file_name": p.name,
                "actual_stem": p.stem,
                "actual_group": group,
                "actual_size_bytes": p.stat().st_size,
            }
        )
    return pd.DataFrame(rows)


def normalize_official_name(name: str) -> str:
    if name.endswith(".h5ad.gz"):
        return name[: -len(".gz")]
    return name


def classify_candidate(row: pd.Series) -> tuple[int, str]:
    group = str(row.get("actual_group", ""))
    name = str(row.get("actual_stem", ""))
    n_obs = row.get("n_obs")
    perturbation_type = str(row.get("perturbation_type", ""))
    modality = str(row.get("modality", ""))
    has_cell_type = bool(row.get("has_cell_type", False))
    has_donor = bool(row.get("has_donor", False))
    suitable_pg = bool(row.get("suitable_perturbation_generalization", False))
    suitable_combo = bool(row.get("suitable_combination_task", False))

    score = 0
    reasons = []
    if group in {"extra_official", "official_generalization"}:
        score += 3
        reasons.append("scPerturBench generalization asset")
    if suitable_pg:
        score += 2
        reasons.append("perturbation-generalization suitable")
    if suitable_combo:
        score += 1
        reasons.append("combination task possible")
    if has_cell_type or has_donor:
        score += 1
        reasons.append("contains context/donor/cell-type metadata")
    if perturbation_type.startswith("chemical"):
        score += 1
        reasons.append("chemical external stress-test")
    if modality == "RNA":
        score += 1
        reasons.append("RNA expression available")
    if pd.notna(n_obs) and n_obs >= 10000:
        score += 1
        reasons.append("enough cells for task-level aggregation")
    if any(key.lower() in name.lower() for key in ["kang", "tcdd", "sciplex", "cross", "frangieh"]):
        score += 2
        reasons.append("high-value external validation target")
    return score, "; ".join(reasons)


def build_asset_tables() -> dict[str, pd.DataFrame]:
    actual = scan_actual_files()
    legacy = pd.read_csv(LEGACY_INVENTORY)
    legacy = legacy.rename(columns={"local_path": "legacy_local_path"})
    legacy["actual_file_name"] = legacy["legacy_local_path"].map(lambda x: Path(str(x)).name)
    merged = actual.merge(legacy, on="actual_file_name", how="left")

    official = pd.concat(
        [
            load_zenodo(SCPERTURBENCH_ZENODO, "scPerturBench Zenodo 14607156"),
            load_zenodo(SCPERTURB_ZENODO, "scPerturb Zenodo 13350497"),
        ],
        ignore_index=True,
    )
    official["normalized_h5ad_name"] = official["file_name"].map(normalize_official_name)
    official["official_is_gzip"] = official["file_name"].str.endswith(".gz")
    coverage = official.merge(
        actual[["actual_file_name", "relative_path", "actual_size_bytes", "actual_group"]],
        left_on="normalized_h5ad_name",
        right_on="actual_file_name",
        how="left",
    )
    coverage["present_on_server"] = coverage["actual_file_name"].notna()
    coverage["size_check_status"] = "not_checked_gzip_source"
    plain = coverage["present_on_server"] & ~coverage["official_is_gzip"]
    coverage.loc[plain, "size_check_status"] = coverage.loc[plain].apply(
        lambda r: "size_match"
        if int(r["official_size_bytes"]) == int(r["actual_size_bytes"])
        else "size_mismatch",
        axis=1,
    )
    coverage.loc[~coverage["present_on_server"], "size_check_status"] = "missing"

    candidate = merged.copy()
    scores = candidate.apply(classify_candidate, axis=1, result_type="expand")
    candidate["e10_priority_score"] = scores[0]
    candidate["priority_reason"] = scores[1]
    candidate["recommended_role"] = "background_asset"
    candidate.loc[candidate["e10_priority_score"] >= 8, "recommended_role"] = "E10_primary_candidate"
    candidate.loc[candidate["e10_priority_score"].between(6, 7), "recommended_role"] = "E10_secondary_candidate"
    candidate = candidate.sort_values(
        ["e10_priority_score", "actual_size_bytes"], ascending=[False, False]
    )

    missing = coverage[~coverage["present_on_server"]].copy()
    return {
        "actual_files": actual,
        "official_coverage": coverage,
        "candidate_ranking": candidate,
        "missing_official_files": missing,
    }


def write_download_manifest(missing: pd.DataFrame) -> None:
    sh = DOWNLOADS / "download_missing_official_files.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "# Generated by run_e10_external_dataset_asset_audit.py",
        "# Large files are intentionally not downloaded automatically by the audit.",
        f'DEST_ROOT="{DATA_ROOT}"',
        "mkdir -p \"$DEST_ROOT\"",
        "",
    ]
    for _, r in missing.iterrows():
        source = r["source"]
        name = r["file_name"]
        url = r["download_url"]
        if not url:
            continue
        if "scPerturBench" in source:
            sub = "extra_official/cellular_context_generalization"
        else:
            sub = "official_scperturb"
        lines.append(f"# {source} | {name} | {r['official_size_bytes']} bytes | {r['official_checksum']}")
        lines.append(f'mkdir -p "$DEST_ROOT/{sub}"')
        lines.append(f'wget --continue -O "$DEST_ROOT/{sub}/{name}" "{url}"')
        if str(name).endswith(".gz"):
            lines.append(f'gunzip -k "$DEST_ROOT/{sub}/{name}"')
        lines.append("")
    while lines and lines[-1] == "":
        lines.pop()
    sh.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sh.chmod(0o755)


def write_report(tables: dict[str, pd.DataFrame]) -> None:
    actual = tables["actual_files"]
    coverage = tables["official_coverage"]
    candidates = tables["candidate_ranking"]
    missing = tables["missing_official_files"]
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    total_size = actual["actual_size_bytes"].sum()
    present = int(coverage["present_on_server"].sum())
    official_total = len(coverage)
    gzip_sources = int(coverage["official_is_gzip"].sum())
    plain_size_match = int((coverage["size_check_status"] == "size_match").sum())
    primary = candidates[candidates["recommended_role"].eq("E10_primary_candidate")].head(12)

    report = f"""# E10 外部任务级验证资产审计

生成时间：{now}

## 1. 当前数据现实

- 实际数据根目录：`{DATA_ROOT}`
- 服务器已有 h5ad：{len(actual)} 个
- 已有 h5ad 总大小：{total_size / 1024**3:.1f} GiB
- 官方 Zenodo 文件覆盖：{present}/{official_total}
- scPerturBench 官方文件为 `.h5ad.gz`：{gzip_sources} 个；服务器保存为解压后 `.h5ad`，大小不能直接与 gzip 包比较。
- scPerturb 官方非压缩 h5ad 大小匹配：{plain_size_match} 个。

结论：E10 不需要盲目重新下载全量数据。当前服务器已经具备外部任务级验证的数据基础。下一步应从候选数据中选择 1–3 个冻结任务级外部验证，而不是重复下载 TB 级数据。

路径注意：当前真实数据根目录是 `{DATA_ROOT}`。为了兼容历史脚本，服务器上已建立本地软链接 `/home/yyf/datasets -> /home/yyf/data`；该软链接不属于 Git 仓库，新机器需要重新建立或直接传入真实数据根目录。

## 2. 推荐 E10 第一批候选

{primary[["actual_file_name","actual_group","study_family","perturbation_type","modality","n_obs","n_vars","e10_priority_score","priority_reason"]].to_markdown(index=False) if False else ""}

| 文件 | 分组 | study | perturbation | modality | cells | genes | score | reason |
|---|---|---|---|---|---:|---:|---:|---|
"""
    for _, r in primary.iterrows():
        report += (
            f"| `{r['actual_file_name']}` | {r.get('actual_group','')} | {r.get('study_family','')} | "
            f"{r.get('perturbation_type','')} | {r.get('modality','')} | {int(r.get('n_obs',0) or 0)} | "
            f"{int(r.get('n_vars',0) or 0)} | {int(r['e10_priority_score'])} | {r['priority_reason']} |\n"
        )

    report += f"""

## 3. 当前缺失文件

官方 metadata 中未在 `{DATA_ROOT}` 找到的文件数：{len(missing)}。

如果后续确实需要补齐，使用：

```bash
bash docs/实验结果/E10_external_task_validation_assets_20260707/download_manifests/download_missing_official_files.sh
```

## 4. 下一步实验建议

1. 首选 `kangCrossCell` / `kangCrossPatient`：数据量适中、外部泛化场景清楚，适合做 E10 task-level validation。
2. 次选 `TCDD` / `sciplex3`：chemical stress-test，可和 Tahoe chemical 边界互相印证。
3. 保留 `Frangieh`：已有 E8b 聚合证据，可用于连接外部 benchmark method-error association 与任务级审计。
4. 不建议先跑 Replogle 大文件：资源消耗高，且不一定立刻提升论文主线。

## 5. 输出文件

- `tables/E10_ACTUAL_H5AD_FILES.csv`
- `tables/E10_OFFICIAL_FILE_COVERAGE.csv`
- `tables/E10_CANDIDATE_RANKING.csv`
- `tables/E10_MISSING_OFFICIAL_FILES.csv`
- `download_manifests/download_missing_official_files.sh`
"""
    (REPORTS / "E10_EXTERNAL_ASSET_AUDIT_REPORT.md").write_text(report, encoding="utf-8")

    def table_html(df: pd.DataFrame, cols: list[str], n: int = 20) -> str:
        return df[cols].head(n).to_html(index=False, escape=True)

    html_doc = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>E10 外部数据资产审计</title>
<style>
body{{margin:0;background:#f7f8f6;color:#17202a;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,"Noto Sans SC",sans-serif;line-height:1.7}}
.top{{background:#12312b;color:white;padding:28px 42px}}.wrap{{max-width:1160px;margin:0 auto;padding:28px}}
.card{{background:white;border:1px solid #d8e0dc;border-radius:16px;padding:22px;margin:18px 0;box-shadow:0 8px 22px rgba(15,23,42,.06);overflow-x:auto}}
h2{{border-bottom:3px solid #0f766e;padding-bottom:8px}}.ok{{border-left:6px solid #047857;background:#f1fbf6}}.warn{{border-left:6px solid #b45309;background:#fff8eb}}
table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border:1px solid #dbe3df;padding:7px;vertical-align:top}}th{{background:#eef6f3}}
code{{background:#eef2f7;padding:2px 5px;border-radius:5px}}
</style></head><body>
<div class="top"><h1>E10 外部任务级验证资产审计</h1><p>先把数据源、文件覆盖和可跑候选钉死，再启动外部任务级验证。</p></div>
<div class="wrap">
<div class="card ok"><h2>当前结论</h2><p>服务器已有 {len(actual)} 个 h5ad，总计 {total_size / 1024**3:.1f} GiB；官方 Zenodo 文件覆盖 {present}/{official_total}。E10 可以直接从现有资产中选候选，不需要全量重下。</p></div>
<div class="card warn"><h2>路径边界</h2><p>旧 inventory 记录的是 <code>/home/yyf/datasets/...</code>，当前真实数据根目录是 <code>{DATA_ROOT}</code>。本服务器已建立兼容软链接 <code>/home/yyf/datasets -&gt; /home/yyf/data</code>；新机器需要重建软链接或直接传入真实根目录。</p></div>
<div class="card"><h2>E10 候选排序</h2>{table_html(candidates, ["actual_file_name","actual_group","study_family","perturbation_type","modality","n_obs","n_vars","e10_priority_score","recommended_role","priority_reason"], 25)}</div>
<div class="card"><h2>官方文件覆盖</h2>{table_html(coverage, ["source","file_name","official_size_bytes","official_checksum","present_on_server","relative_path","size_check_status"], 40)}</div>
<div class="card"><h2>缺失官方文件</h2>{table_html(missing, ["source","file_name","official_size_bytes","official_checksum","download_url"], 40)}</div>
</div></body></html>
"""
    (REPORTS / "E10_EXTERNAL_ASSET_AUDIT.html").write_text(html_doc, encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    ensure_metadata()

    tables = build_asset_tables()
    tables["actual_files"].to_csv(TABLES / "E10_ACTUAL_H5AD_FILES.csv", index=False)
    tables["official_coverage"].to_csv(TABLES / "E10_OFFICIAL_FILE_COVERAGE.csv", index=False)
    tables["candidate_ranking"].to_csv(TABLES / "E10_CANDIDATE_RANKING.csv", index=False)
    tables["missing_official_files"].to_csv(TABLES / "E10_MISSING_OFFICIAL_FILES.csv", index=False)
    write_download_manifest(tables["missing_official_files"])
    write_report(tables)

    status = {
        "status": "ok",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "output_dir": str(OUT.relative_to(ROOT)),
        "data_root": str(DATA_ROOT),
        "input_git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout.strip(),
        "source_files": {
            "legacy_inventory": str(LEGACY_INVENTORY.relative_to(ROOT)),
            "scperturbench_zenodo": str(SCPERTURBENCH_ZENODO.relative_to(ROOT)),
            "scperturb_zenodo": str(SCPERTURB_ZENODO.relative_to(ROOT)),
            "scperturbench_readme": str(SCPERTURBENCH_README.relative_to(ROOT)),
            "scperturbench_md5_xlsx": str(SCPERTURBENCH_MD5_XLSX.relative_to(ROOT)),
        },
    }
    (OUT / "RUN_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    readme = """# E10 external task-level validation assets

入口：

- `reports/E10_EXTERNAL_ASSET_AUDIT.html`
- `reports/E10_EXTERNAL_ASSET_AUDIT_REPORT.md`

运行命令：

```bash
python3 tools/scripts/run_e10_external_dataset_asset_audit.py
```
"""
    (OUT / "README_先看这个.md").write_text(readme, encoding="utf-8")
    print(f"Wrote E10 asset audit to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
