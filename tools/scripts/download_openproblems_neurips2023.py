#!/usr/bin/env python3
"""Download OpenProblems / NeurIPS 2023 single-cell perturbation resources.

This dataset is useful as a non-Tahoe chemical perturbation panel:
PBMCs, 144 small molecules, multiple donors/cell types, plus a raw
single-cell layer that can later support richer context/generalization checks.

The script uses public HTTPS mirrors for the official S3 objects, starts
aria2c in the background, and writes status/manifest files beside the data.
Re-running is safe because aria2c resumes partial downloads.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path(
    "/home/yyf/data/singlecell_perturbation_atlas/mega_external/"
    "OpenProblems_NeurIPS2023_single_cell_perturbations"
)
DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "download_logs"

FILES = [
    {
        "relative_path": "processed/neurips-2023-data/de_train.h5ad",
        "url": (
            "https://openproblems-data.s3.amazonaws.com/resources/"
            "task_perturbation_prediction/datasets/neurips-2023-data/de_train.h5ad"
        ),
        "expected_bytes": 183_168_750,
        "description": "OpenProblems processed differential-expression training matrix.",
    },
    {
        "relative_path": "processed/neurips-2023-data/de_test.h5ad",
        "url": (
            "https://openproblems-data.s3.amazonaws.com/resources/"
            "task_perturbation_prediction/datasets/neurips-2023-data/de_test.h5ad"
        ),
        "expected_bytes": 109_139_040,
        "description": "OpenProblems processed differential-expression held-out matrix.",
    },
    {
        "relative_path": "processed/neurips-2023-data/id_map.csv",
        "url": (
            "https://openproblems-data.s3.amazonaws.com/resources/"
            "task_perturbation_prediction/datasets/neurips-2023-data/id_map.csv"
        ),
        "expected_bytes": 3_860,
        "description": "OpenProblems test-row mapping table.",
    },
    {
        "relative_path": "kaggle_original/2023-09-12_de_by_cell_type_train.h5ad",
        "url": (
            "https://openproblems-bio.s3.amazonaws.com/public/neurips-2023-competition/"
            "2023-09-14_kaggle_upload/2023-09-12_de_by_cell_type_train.h5ad"
        ),
        "expected_bytes": 537_565_848,
        "description": "Original Kaggle train DGE file from the NeurIPS 2023 challenge.",
    },
    {
        "relative_path": "kaggle_original/2023-09-12_de_by_cell_type_test.h5ad",
        "url": (
            "https://openproblems-bio.s3.amazonaws.com/public/neurips-2023-competition/"
            "2023-09-14_kaggle_upload/2023-09-12_de_by_cell_type_test.h5ad"
        ),
        "expected_bytes": 223_729_720,
        "description": "Original Kaggle test DGE file from the NeurIPS 2023 challenge.",
    },
    {
        "relative_path": "raw/sc_counts_reannotated_with_counts.h5ad",
        "url": (
            "https://openproblems-bio.s3.amazonaws.com/public/neurips-2023-competition/"
            "sc_counts_reannotated_with_counts.h5ad"
        ),
        "expected_bytes": 3_515_260_667,
        "description": "Raw reannotated single-cell counts for later cell-level analyses.",
    },
]

# Additional public S3 objects found under
# s3://openproblems-bio/public/neurips-2023-competition/.
# These make the dataset useful beyond the Kaggle DGE task: multiome,
# mechanism-of-action annotations, pseudobulk layers, and workflow resources.
FILES.extend(
    [
        {
            "relative_path": "kaggle_original/2023-08-31_sc_multiome_expression_atac.h5ad",
            "url": (
                "https://openproblems-bio.s3.amazonaws.com/public/neurips-2023-competition/"
                "2023-09-14_kaggle_upload/2023-08-31_sc_multiome_expression_atac.h5ad"
            ),
            "expected_bytes": 3_490_212_284,
            "description": "Kaggle baseline multiome expression/ATAC file.",
        },
        {
            "relative_path": "kaggle_original/2023-09-14_sc_expression_train.h5ad",
            "url": (
                "https://openproblems-bio.s3.amazonaws.com/public/neurips-2023-competition/"
                "2023-09-14_kaggle_upload/2023-09-14_sc_expression_train.h5ad"
            ),
            "expected_bytes": 9_926_648_044,
            "description": "Original Kaggle single-cell expression training file.",
        },
        {
            "relative_path": "raw/de_per_plate.h5ad",
            "url": "https://openproblems-bio.s3.amazonaws.com/public/neurips-2023-competition/de_per_plate.h5ad",
            "expected_bytes": 647_678_284,
            "description": "Differential-expression layer aggregated by plate.",
        },
        {
            "relative_path": "raw/de_per_plate_by_cell_type.h5ad",
            "url": (
                "https://openproblems-bio.s3.amazonaws.com/public/neurips-2023-competition/"
                "de_per_plate_by_cell_type.h5ad"
            ),
            "expected_bytes": 3_767_020_502,
            "description": "Differential-expression layer aggregated by plate and cell type.",
        },
        {
            "relative_path": "metadata/moa_annotations.csv",
            "url": "https://openproblems-bio.s3.amazonaws.com/public/neurips-2023-competition/moa_annotations.csv",
            "expected_bytes": 23_824,
            "description": "Mechanism-of-action annotations for small molecules.",
        },
        {
            "relative_path": "multiome/multiome_counts.h5mu",
            "url": "https://openproblems-bio.s3.amazonaws.com/public/neurips-2023-competition/multiome_counts.h5mu",
            "expected_bytes": 3_954_049_705,
            "description": "Raw multiome counts.",
        },
        {
            "relative_path": "multiome/multiome_counts_processed.h5mu",
            "url": (
                "https://openproblems-bio.s3.amazonaws.com/public/neurips-2023-competition/"
                "multiome_counts_processed.h5mu"
            ),
            "expected_bytes": 3_348_743_937,
            "description": "Processed multiome counts.",
        },
        {
            "relative_path": "multiome/multiome_counts_reannotated.h5mu",
            "url": (
                "https://openproblems-bio.s3.amazonaws.com/public/neurips-2023-competition/"
                "multiome_counts_reannotated.h5mu"
            ),
            "expected_bytes": 3_348_743_937,
            "description": "Reannotated multiome counts.",
        },
        {
            "relative_path": "raw/pseudobulk.h5ad",
            "url": "https://openproblems-bio.s3.amazonaws.com/public/neurips-2023-competition/pseudobulk.h5ad",
            "expected_bytes": 168_692_752,
            "description": "Pseudobulk layer.",
        },
        {
            "relative_path": "raw/pseudobulk_by_cell_type.h5ad",
            "url": (
                "https://openproblems-bio.s3.amazonaws.com/public/neurips-2023-competition/"
                "pseudobulk_by_cell_type.h5ad"
            ),
            "expected_bytes": 967_858_190,
            "description": "Pseudobulk layer grouped by cell type.",
        },
        {
            "relative_path": "raw/sc_counts.h5ad",
            "url": "https://openproblems-bio.s3.amazonaws.com/public/neurips-2023-competition/sc_counts.h5ad",
            "expected_bytes": 14_258_126_612,
            "description": "Original single-cell counts.",
        },
        {
            "relative_path": "raw/sc_counts_processed.h5ad",
            "url": "https://openproblems-bio.s3.amazonaws.com/public/neurips-2023-competition/sc_counts_processed.h5ad",
            "expected_bytes": 3_515_260_667,
            "description": "Processed single-cell counts.",
        },
        {
            "relative_path": "workflow_resources/neurips-2023-data/dataset_info.yaml",
            "url": (
                "https://openproblems-bio.s3.amazonaws.com/public/neurips-2023-competition/"
                "workflow-resources/neurips-2023-data/dataset_info.yaml"
            ),
            "expected_bytes": 1_670,
            "description": "OpenProblems workflow dataset metadata.",
        },
        {
            "relative_path": "workflow_resources/neurips-2023-data/de_test.h5ad",
            "url": (
                "https://openproblems-bio.s3.amazonaws.com/public/neurips-2023-competition/"
                "workflow-resources/neurips-2023-data/de_test.h5ad"
            ),
            "expected_bytes": 109_136_932,
            "description": "Workflow copy of processed DE test file.",
        },
        {
            "relative_path": "workflow_resources/neurips-2023-data/de_train.h5ad",
            "url": (
                "https://openproblems-bio.s3.amazonaws.com/public/neurips-2023-competition/"
                "workflow-resources/neurips-2023-data/de_train.h5ad"
            ),
            "expected_bytes": 183_170_148,
            "description": "Workflow copy of processed DE train file.",
        },
        {
            "relative_path": "workflow_resources/neurips-2023-data/id_map.csv",
            "url": (
                "https://openproblems-bio.s3.amazonaws.com/public/neurips-2023-competition/"
                "workflow-resources/neurips-2023-data/id_map.csv"
            ),
            "expected_bytes": 3_860,
            "description": "Workflow id map.",
        },
        {
            "relative_path": "workflow_resources/neurips-2023-data/prediction.h5ad",
            "url": (
                "https://openproblems-bio.s3.amazonaws.com/public/neurips-2023-competition/"
                "workflow-resources/neurips-2023-data/prediction.h5ad"
            ),
            "expected_bytes": 3_687_489,
            "description": "Workflow reference prediction artifact.",
        },
        {
            "relative_path": "workflow_resources/neurips-2023-data/pseudobulk.h5ad",
            "url": (
                "https://openproblems-bio.s3.amazonaws.com/public/neurips-2023-competition/"
                "workflow-resources/neurips-2023-data/pseudobulk.h5ad"
            ),
            "expected_bytes": 141_436_419,
            "description": "Workflow pseudobulk artifact.",
        },
        {
            "relative_path": "workflow_resources/neurips-2023-data/score.h5ad",
            "url": (
                "https://openproblems-bio.s3.amazonaws.com/public/neurips-2023-competition/"
                "workflow-resources/neurips-2023-data/score.h5ad"
            ),
            "expected_bytes": 22_264,
            "description": "Workflow score artifact.",
        },
        {
            "relative_path": "workflow_resources/neurips-2023-data/state.yaml",
            "url": (
                "https://openproblems-bio.s3.amazonaws.com/public/neurips-2023-competition/"
                "workflow-resources/neurips-2023-data/state.yaml"
            ),
            "expected_bytes": 116,
            "description": "Workflow state file.",
        },
        {
            "relative_path": "workflow_resources/neurips-2023-kaggle/2023-09-12_de_by_cell_type_test.h5ad",
            "url": (
                "https://openproblems-bio.s3.amazonaws.com/public/neurips-2023-competition/"
                "workflow-resources/neurips-2023-kaggle/2023-09-12_de_by_cell_type_test.h5ad"
            ),
            "expected_bytes": 130_812_514,
            "description": "Workflow Kaggle DE test file.",
        },
        {
            "relative_path": "workflow_resources/neurips-2023-kaggle/2023-09-12_de_by_cell_type_train.h5ad",
            "url": (
                "https://openproblems-bio.s3.amazonaws.com/public/neurips-2023-competition/"
                "workflow-resources/neurips-2023-kaggle/2023-09-12_de_by_cell_type_train.h5ad"
            ),
            "expected_bytes": 295_742_055,
            "description": "Workflow Kaggle DE train file.",
        },
        {
            "relative_path": "workflow_resources/neurips-2023-kaggle/dataset_info.yaml",
            "url": (
                "https://openproblems-bio.s3.amazonaws.com/public/neurips-2023-competition/"
                "workflow-resources/neurips-2023-kaggle/dataset_info.yaml"
            ),
            "expected_bytes": 1_681,
            "description": "Workflow Kaggle dataset metadata.",
        },
        {
            "relative_path": "workflow_resources/neurips-2023-kaggle/de_test.h5ad",
            "url": (
                "https://openproblems-bio.s3.amazonaws.com/public/neurips-2023-competition/"
                "workflow-resources/neurips-2023-kaggle/de_test.h5ad"
            ),
            "expected_bytes": 181_666_552,
            "description": "Workflow converted Kaggle DE test file.",
        },
        {
            "relative_path": "workflow_resources/neurips-2023-kaggle/de_train.h5ad",
            "url": (
                "https://openproblems-bio.s3.amazonaws.com/public/neurips-2023-competition/"
                "workflow-resources/neurips-2023-kaggle/de_train.h5ad"
            ),
            "expected_bytes": 391_078_222,
            "description": "Workflow converted Kaggle DE train file.",
        },
        {
            "relative_path": "workflow_resources/neurips-2023-kaggle/id_map.csv",
            "url": (
                "https://openproblems-bio.s3.amazonaws.com/public/neurips-2023-competition/"
                "workflow-resources/neurips-2023-kaggle/id_map.csv"
            ),
            "expected_bytes": 6_723,
            "description": "Workflow Kaggle id map.",
        },
        {
            "relative_path": "workflow_resources/neurips-2023-kaggle/prediction.h5ad",
            "url": (
                "https://openproblems-bio.s3.amazonaws.com/public/neurips-2023-competition/"
                "workflow-resources/neurips-2023-kaggle/prediction.h5ad"
            ),
            "expected_bytes": 22_440_046,
            "description": "Workflow Kaggle reference prediction artifact.",
        },
        {
            "relative_path": "workflow_resources/neurips-2023-kaggle/score.h5ad",
            "url": (
                "https://openproblems-bio.s3.amazonaws.com/public/neurips-2023-competition/"
                "workflow-resources/neurips-2023-kaggle/score.h5ad"
            ),
            "expected_bytes": 22_264,
            "description": "Workflow Kaggle score artifact.",
        },
        {
            "relative_path": "workflow_resources/neurips-2023-kaggle/state.yaml",
            "url": (
                "https://openproblems-bio.s3.amazonaws.com/public/neurips-2023-competition/"
                "workflow-resources/neurips-2023-kaggle/state.yaml"
            ),
            "expected_bytes": 117,
            "description": "Workflow Kaggle state file.",
        },
    ]
)


def local_path(item: dict) -> Path:
    return DATA_DIR / item["relative_path"]


def container_integrity(path: Path) -> tuple[bool | None, str]:
    """Check that completed HDF5 containers can read their root object table.

    Exact byte count is insufficient for an interrupted aria2 transfer: a
    sparse target can already have its final apparent size while internal HDF5
    blocks are still absent.  CSV/YAML files are left as not-applicable.
    """
    if path.suffix.lower() not in {".h5ad", ".h5mu", ".h5"}:
        return None, "not_applicable"
    if not path.exists():
        return False, "missing"
    try:
        import h5py
        with h5py.File(path, "r") as handle:
            _ = list(handle.keys())
        return True, "hdf5_root_readable"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def file_state(item: dict) -> dict:
    path = local_path(item)
    partial = Path(str(path) + ".aria2")
    size = path.stat().st_size if path.exists() else 0
    disk_bytes = path.stat().st_blocks * 512 if path.exists() else 0
    expected = item["expected_bytes"]
    size_complete = path.exists() and size == expected and not partial.exists()
    integrity_ok, integrity_detail = container_integrity(path) if size_complete else (None, "not_checked_until_size_complete")
    complete = size_complete and integrity_ok is not False
    return {
        "relative_path": item["relative_path"],
        "path": str(path),
        "exists": path.exists(),
        "partial": partial.exists(),
        "apparent_bytes": size,
        "disk_bytes": disk_bytes,
        "expected_bytes": expected,
        "complete": complete,
        "container_integrity_ok": integrity_ok,
        "container_integrity_detail": integrity_detail,
        "description": item["description"],
    }


def status() -> dict:
    states = [file_state(item) for item in FILES]
    return {
        "dataset": "OpenProblems / NeurIPS 2023 single-cell perturbations",
        "root": str(ROOT),
        "data_dir": str(DATA_DIR),
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "n_files": len(states),
        "n_complete": sum(1 for s in states if s["complete"]),
        "n_partial": sum(1 for s in states if s["partial"]),
        "apparent_size_gb": round(sum(s["apparent_bytes"] for s in states) / 1e9, 3),
        "disk_usage_gb": round(sum(s["disk_bytes"] for s in states) / 1e9, 3),
        "expected_size_gb": round(sum(s["expected_bytes"] for s in states) / 1e9, 3),
        "files": states,
    }


def build_aria2_input(path: Path, include_existing: bool = False) -> list[str]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    selected: list[str] = []
    lines: list[str] = []
    for item in FILES:
        state = file_state(item)
        if state["complete"] and not include_existing:
            continue
        out_path = Path(item["relative_path"])
        (DATA_DIR / out_path.parent).mkdir(parents=True, exist_ok=True)
        selected.append(item["relative_path"])
        # The openproblems-bio S3 HTTPS endpoint intermittently fails in this
        # network with TLS EOF.  The same public S3 objects are readable over
        # HTTP, so use HTTP for that host while keeping the openproblems-data
        # endpoint on HTTPS.
        url = item["url"].replace("https://openproblems-bio.s3.amazonaws.com/", "http://openproblems-bio.s3.amazonaws.com/")
        lines.append(url)
        lines.append(f"  out={item['relative_path']}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["status", "manifest", "start"], default="start")
    parser.add_argument("--concurrent", type=int, default=4)
    parser.add_argument("--split", type=int, default=4)
    parser.add_argument("--all", action="store_true", help="Include already-complete files.")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    status_path = LOG_DIR / "OPENPROBLEMS_NEURIPS2023_DOWNLOAD_STATUS.json"

    if args.mode == "status":
        stat = status()
        status_path.write_text(json.dumps(stat, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(stat, ensure_ascii=False, indent=2))
        return

    input_path = LOG_DIR / "openproblems_neurips2023_aria2_input.txt"
    selected = build_aria2_input(input_path, include_existing=args.all)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_repository": "https://github.com/openproblems-bio/task_perturbation_prediction",
        "competition": "https://www.kaggle.com/competitions/open-problems-single-cell-perturbations/overview",
        "paper": "https://openreview.net/forum?id=WTI4RJYSVm",
        "n_expected_files": len(FILES),
        "n_selected_for_download": len(selected),
        "selected": selected,
        "data_dir": str(DATA_DIR),
        "aria2_input": str(input_path),
        "files": FILES,
    }
    (LOG_DIR / "OPENPROBLEMS_NEURIPS2023_DOWNLOAD_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

    if args.mode == "manifest":
        return
    if not selected:
        print("[done] all files already complete")
        return

    log_file = LOG_DIR / "openproblems_neurips2023_aria2.log"
    console_file = LOG_DIR / "openproblems_neurips2023_aria2.console.log"
    cmd = [
        "aria2c",
        "--continue=true",
        "--auto-file-renaming=false",
        "--allow-overwrite=false",
        f"--max-concurrent-downloads={args.concurrent}",
        f"--split={args.split}",
        f"--max-connection-per-server={args.split}",
        "--min-split-size=8M",
        "--max-tries=30",
        "--retry-wait=15",
        "--summary-interval=30",
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
        **status(),
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
