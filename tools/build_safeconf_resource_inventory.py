#!/usr/bin/env python3
"""Build compact file inventories for SafeConf server data and outputs."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def _iso_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def _write_file_inventory(root: Path, target: Path) -> tuple[int, int]:
    count = 0
    total = 0
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "relative_path",
                "size_bytes",
                "modified_utc",
                "suffix",
            ],
        )
        writer.writeheader()
        for path in _iter_files(root):
            stat = path.stat()
            writer.writerow(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "size_bytes": stat.st_size,
                    "modified_utc": _iso_mtime(path),
                    "suffix": path.suffix.lower(),
                }
            )
            count += 1
            total += stat.st_size
    return count, total


def _write_output_summary(root: Path, target: Path) -> list[dict]:
    rows: list[dict] = []
    for child in sorted(path for path in root.iterdir() if path.is_dir()):
        files = list(_iter_files(child))
        rows.append(
            {
                "output_directory": child.name,
                "file_count": len(files),
                "size_bytes": sum(path.stat().st_size for path in files),
                "latest_modified_utc": max(
                    (_iso_mtime(path) for path in files),
                    default="",
                ),
            }
        )
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [
            "output_directory",
            "file_count",
            "size_bytes",
            "latest_modified_utc",
        ])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _suffix_summary(root: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"files": 0, "bytes": 0})
    for path in _iter_files(root):
        key = path.suffix.lower() or "[no_suffix]"
        counts[key]["files"] += 1
        counts[key]["bytes"] += path.stat().st_size
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    outputs_root = args.outputs_root.resolve()
    out_dir = args.out_dir.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    if not outputs_root.is_dir():
        raise FileNotFoundError(outputs_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_count, data_bytes = _write_file_inventory(
        data_root, out_dir / "SERVER_DATA_FILE_INVENTORY.csv"
    )
    output_count, output_bytes = _write_file_inventory(
        outputs_root, out_dir / "SERVER_OUTPUT_FILE_INVENTORY.csv"
    )
    output_dirs = _write_output_summary(
        outputs_root, out_dir / "SERVER_OUTPUT_DIRECTORY_SUMMARY.csv"
    )
    summary = {
        "generated_utc": datetime.now(tz=timezone.utc).isoformat(),
        "host": platform.node(),
        "data_root": str(data_root),
        "outputs_root": str(outputs_root),
        "data_file_count": data_count,
        "data_size_bytes": data_bytes,
        "output_file_count": output_count,
        "output_size_bytes": output_bytes,
        "output_directory_count": len(output_dirs),
        "data_suffix_summary": _suffix_summary(data_root),
        "output_suffix_summary": _suffix_summary(outputs_root),
    }
    (out_dir / "SERVER_RESOURCE_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
