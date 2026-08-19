#!/usr/bin/env python3
"""Resumable acquisition of the official X-Atlas/Orion processed release.

X-Atlas/Orion is an external, genome-wide Perturb-seq data line complementary
to the local scPerturBench/scPerturb collection and Tahoe.  It contains two
human cell-line screens (HCT116 and HEK293T) with dual guides targeting the
same gene.  The files are large, so this downloader records the official
Figshare source URLs and resumes with aria2c instead of duplicating files.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path("/home/yyf/data/singlecell_perturbation_atlas/mega_external/X_Atlas_Orion_2025")
DATA = ROOT / "data"
LOGS = ROOT / "download_logs"
FILES = [
    {"name": "HCT116_filtered_dual_guide_cells.h5ad.md5", "url": "https://ndownloader.figshare.com/files/55021208", "bytes": 81, "role": "Official MD5 checksum for HCT116 h5ad."},
    {"name": "HCT116_filtered_dual_guide_cells.h5ad", "url": "https://ndownloader.figshare.com/files/55021257", "bytes": 209_354_246_272, "role": "Genome-wide Perturb-seq HCT116 processed cells."},
    {"name": "HEK293T_filtered_dual_guide_cells.h5ad.md5", "url": "https://ndownloader.figshare.com/files/55025077", "bytes": 82, "role": "Official MD5 checksum for HEK293T h5ad."},
    {"name": "HEK293T_filtered_dual_guide_cells.h5ad", "url": "https://ndownloader.figshare.com/files/55074802", "bytes": 350_164_035_901, "role": "Genome-wide Perturb-seq HEK293T processed cells."},
    {"name": "guide_library.csv", "url": "https://ndownloader.figshare.com/files/57368587", "bytes": 3_648_784, "role": "Guide/library annotation metadata."},
    {"name": "HCT116_filtered_guide_calls_per_cell.csv.gz", "url": "https://ndownloader.figshare.com/files/59490731", "bytes": 69_606_079, "role": "HCT116 guide-call metadata."},
    {"name": "HEK293T_filtered_guide_calls_per_cell.csv.gz", "url": "https://ndownloader.figshare.com/files/59490734", "bytes": 90_886_568, "role": "HEK293T guide-call metadata."},
]
ARTICLE_URL = "https://figshare.com/articles/dataset/29190726"


def status() -> dict:
    DATA.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in FILES:
        path = DATA / item["name"]
        partial = Path(str(path) + ".aria2").exists()
        n = path.stat().st_size if path.exists() else 0
        rows.append({**item, "exists": path.exists(), "partial": partial, "apparent_bytes": n, "complete": bool(path.exists() and not partial and n == item["bytes"])})
    return {
        "dataset": "X-Atlas/Orion official processed Perturb-seq release",
        "article_url": ARTICLE_URL,
        "data_dir": str(DATA),
        "expected_size_gb": round(sum(x["bytes"] for x in FILES) / 1e9, 3),
        "downloaded_size_gb": round(sum(x["apparent_bytes"] for x in rows) / 1e9, 3),
        "n_files": len(rows),
        "n_complete": sum(x["complete"] for x in rows),
        "n_partial": sum(x["partial"] for x in rows),
        "files": rows,
        "checked_at": datetime.now().isoformat(timespec="seconds"),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["status", "manifest", "start"], default="start")
    p.add_argument("--concurrent", type=int, default=2)
    p.add_argument("--split", type=int, default=4)
    args = p.parse_args()
    DATA.mkdir(parents=True, exist_ok=True); LOGS.mkdir(parents=True, exist_ok=True)
    if args.mode == "status":
        s = status(); (LOGS / "XATLAS_ORION_STATUS.json").write_text(json.dumps(s, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); print(json.dumps(s, ensure_ascii=False, indent=2)); return
    selected, lines = [], []
    for item in FILES:
        path = DATA / item["name"]
        if path.exists() and path.stat().st_size == item["bytes"] and not Path(str(path)+".aria2").exists():
            continue
        selected.append(item["name"]); lines += [item["url"], f"  out={item['name']}"]
    inp = LOGS / "xatlas_orion_aria2_input.txt"; inp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    manifest = {"created_at": datetime.now().isoformat(timespec="seconds"), "article_url": ARTICLE_URL, "official_figshare_article_id": 29190726, "data_dir": str(DATA), "n_selected": len(selected), "selected": selected, "expected_total_size_gb": round(sum(x["bytes"] for x in FILES)/1e9,3), "source_note": "Official Figshare files resolved through the Figshare v2 API on 2026-07-11."}
    (LOGS / "XATLAS_ORION_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if args.mode == "manifest" or not selected: return
    log = LOGS / "xatlas_orion_aria2.log"; console = LOGS / "xatlas_orion_aria2.console.log"
    cmd = ["aria2c", "--continue=true", "--auto-file-renaming=false", "--allow-overwrite=false", f"--max-concurrent-downloads={args.concurrent}", f"--split={args.split}", "--max-connection-per-server=4", "--min-split-size=16M", "--max-tries=30", "--retry-wait=30", "--summary-interval=60", "--check-certificate=true", f"--log={log}", "--log-level=notice", f"--dir={DATA}", f"--input-file={inp}"]
    with console.open("ab") as out:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, start_new_session=True, env=os.environ.copy())
    run = {**status(), "started_at": datetime.now().isoformat(timespec="seconds"), "pid": proc.pid, "cmd": cmd, "console_log": str(console), "aria2_log": str(log)}
    (LOGS / "XATLAS_ORION_STATUS.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[started] pid={proc.pid}")


if __name__ == "__main__":
    main()
