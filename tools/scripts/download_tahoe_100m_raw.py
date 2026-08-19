#!/usr/bin/env python3
"""Download Tahoe-100M raw single-cell parquet shards from HuggingFace.

The pseudobulk differential-expression shards are already used by SafeConf.
This downloader fetches the raw `data/train-xxxxx-of-03388.parquet` files so
the project has the larger data layer requested for broader follow-up work.

It starts aria2c in the background by default and writes a PID/log/URL manifest
under the dataset directory.  Re-running is safe: aria2c continues partial
downloads.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path("/home/yyf/data/singlecell_perturbation_atlas/mega_external/Tahoe-100M")
DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "download_logs"
N_SHARDS = 3388
REPO_URL = "https://huggingface.co/datasets/tahoebio/Tahoe-100M/resolve/main"


def shard_name(i: int) -> str:
    return f"train-{i:05d}-of-{N_SHARDS:05d}.parquet"


def shard_url(i: int) -> str:
    return f"{REPO_URL}/data/{shard_name(i)}?download=true"


def build_aria2_input(path: Path, missing_only: bool = True) -> list[str]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    selected: list[str] = []
    lines: list[str] = []
    for i in range(N_SHARDS):
        name = shard_name(i)
        target = DATA_DIR / name
        # aria2 creates both the target file and `<target>.aria2` while the
        # shard is incomplete.  A non-empty target with a sidecar must still be
        # included so a restarted job can continue it.
        if (
            missing_only
            and target.exists()
            and target.stat().st_size > 0
            and not Path(str(target) + ".aria2").exists()
        ):
            continue
        selected.append(name)
        lines.append(shard_url(i))
        lines.append(f"  out={name}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return selected


def count_status() -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    files = list(DATA_DIR.glob("train-*-of-03388.parquet"))
    aria = list(DATA_DIR.glob("*.aria2"))
    size = sum(p.stat().st_size for p in files if p.exists())
    present = set()
    for p in files:
        try:
            present.add(int(p.name.split("-")[1]))
        except Exception:
            pass
    missing = [i for i in range(N_SHARDS) if i not in present]
    complete = [p for p in files if not Path(str(p) + ".aria2").exists()]
    partial = [p for p in files if Path(str(p) + ".aria2").exists()]
    return {
        "data_dir": str(DATA_DIR),
        "complete_parquet_files": len(complete),
        "partial_parquet_files": len(partial),
        "complete_or_partial_parquet_files": len(files),
        "aria2_partial_files": len(aria),
        "expected_shards": N_SHARDS,
        "missing_count_by_filename": len(missing),
        "first_missing": missing[:20],
        "last_missing": missing[-20:],
        "downloaded_size_gb": round(size / 1e9, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["status", "start", "manifest"], default="start")
    parser.add_argument("--concurrent", type=int, default=8)
    parser.add_argument("--split", type=int, default=4)
    parser.add_argument("--all", action="store_true", help="Include existing files in aria2 manifest.")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    status_path = LOG_DIR / "TAHOE_RAW_DOWNLOAD_STATUS.json"

    if args.mode == "status":
        status = count_status()
        status["checked_at"] = datetime.now().isoformat(timespec="seconds")
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return

    input_path = LOG_DIR / "tahoe_100m_raw_aria2_input.txt"
    selected = build_aria2_input(input_path, missing_only=not args.all)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repo": "tahoebio/Tahoe-100M",
        "n_expected_shards": N_SHARDS,
        "n_selected_for_download": len(selected),
        "data_dir": str(DATA_DIR),
        "aria2_input": str(input_path),
        "first_selected": selected[:20],
        "last_selected": selected[-20:],
    }
    (LOG_DIR / "TAHOE_RAW_DOWNLOAD_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

    if args.mode == "manifest":
        return

    if not selected:
        print("[done] no missing shards selected")
        return

    log_file = LOG_DIR / "tahoe_100m_raw_aria2.log"
    console_file = LOG_DIR / "tahoe_100m_raw_aria2.console.log"
    cmd = [
        "aria2c",
        "--continue=true",
        "--auto-file-renaming=false",
        "--allow-overwrite=false",
        f"--max-concurrent-downloads={args.concurrent}",
        f"--split={args.split}",
        "--min-split-size=8M",
        "--max-tries=30",
        "--retry-wait=15",
        "--summary-interval=60",
        "--check-certificate=true",
        f"--log={log_file}",
        "--log-level=notice",
        f"--dir={DATA_DIR}",
        f"--input-file={input_path}",
    ]
    with console_file.open("ab") as out:
        proc = subprocess.Popen(
            cmd,
            stdout=out,
            stderr=subprocess.STDOUT,
            cwd=str(ROOT),
            start_new_session=True,
            env={**os.environ},
        )
    run_status = {
        **count_status(),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "pid": proc.pid,
        "cmd": cmd,
        "console_log": str(console_file),
        "aria2_log": str(log_file),
    }
    status_path.write_text(json.dumps(run_status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[started] pid={proc.pid}")
    print(f"[log] {console_file}")


if __name__ == "__main__":
    main()
