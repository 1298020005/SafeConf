#!/usr/bin/env python3
"""Open frozen RPE1 perturbation rows only after E192 predictions are remote."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
EVALUATOR = ROOT / "tools/scripts/run_e192_adamson_rpe1_evaluation.py"
ASSET_BUILDER = ROOT / "tools/scripts/build_e192_adamson_rpe1_pretruth_assets.py"
OUT = ROOT / "docs/实验结果/E192_adamson_to_replogle_rpe1_locked_transfer_20260729"
DATA = Path("/home/yyf/data/safeconf_e192_adamson_rpe1")
ASSETS = DATA / "model_assets"
TARGET = Path(
    "/home/yyf/data/singlecell_perturbation_atlas/official_generalization/"
    "Replogle_RPE1essential.h5ad"
)
TRUTH = OUT / "evaluation_truth"
N_GENES = 512
N_QUERIES = 175
N_TARGET_ROWS = 1086


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_text(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def require_committed(path: Path, head: str) -> None:
    relative = path.relative_to(ROOT).as_posix()
    committed = subprocess.check_output(["git", "show", f"{head}:{relative}"], cwd=ROOT)
    if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(committed).digest():
        raise RuntimeError(f"working file differs from committed freeze: {relative}")


def verify_prediction_freeze() -> dict[str, Any]:
    head = git_text("rev-parse", "HEAD")
    branch = git_text("branch", "--show-current")
    if not branch:
        raise RuntimeError("E192 truth opening requires a named branch")
    remote_heads: dict[str, str] = {}
    for remote in ("origin", "github"):
        subprocess.run(
            [
                "git",
                "fetch",
                "--quiet",
                remote,
                f"refs/heads/{branch}:refs/remotes/{remote}/{branch}",
            ],
            cwd=ROOT,
            check=True,
        )
        remote_head = git_text("rev-parse", f"refs/remotes/{remote}/{branch}")
        if subprocess.run(
            ["git", "merge-base", "--is-ancestor", head, remote_head],
            cwd=ROOT,
            check=False,
        ).returncode:
            raise RuntimeError(f"HEAD absent from {remote}/{branch}")
        remote_heads[remote] = remote_head
    frozen = [
        RUNNER,
        EVALUATOR,
        ASSET_BUILDER,
        OUT / "PREREG_CONFIRMATION_PLAN.md",
        OUT / "METADATA_FREEZE_STATUS.json",
        OUT / "E192_QUERY_MANIFEST.csv",
        OUT / "E192_TARGET_CELL_TASK_ASSIGNMENTS.csv",
        OUT / "E192_TARGET_CONTROL_ASSIGNMENTS.csv",
        OUT / "RAW_INPUT_LOCKS.json",
        OUT / "MODEL_ASSET_LOCKS.csv",
        OUT / "pretruth_release/PRETRUTH_STATUS.json",
        OUT / "pretruth_release/RELEASE_LOCKS.csv",
        OUT / "pretruth_release/arrays/PRETRUTH_PREDICTIONS.npz",
        OUT / "pretruth_release/tables/QUERY_ORDER.csv",
    ]
    for path in frozen:
        require_committed(path, head)
    status = json.loads((OUT / "pretruth_release/PRETRUTH_STATUS.json").read_text())
    if (
        status.get("status") != "PASS"
        or status.get("target_perturbation_x_rows_read") != 0
        or status.get("target_query_graphs_containing_y") != 0
        or status.get("n_target_queries") != N_QUERIES
    ):
        raise RuntimeError("E192 pretruth status does not authorize truth opening")
    return {
        "git_head": head,
        "git_branch": branch,
        "prediction_freeze_remote_heads": remote_heads,
    }


def import_asset_builder() -> Any:
    spec = importlib.util.spec_from_file_location("e192_asset_builder_for_truth", ASSET_BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen E192 normalization")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {str(key): np.asarray(archive[key], np.float32) for key in archive.files}


def main() -> None:
    if TRUTH.exists():
        raise RuntimeError("E192 target truth release is append-only and already exists")
    freeze = verify_prediction_freeze()
    raw_locks = json.loads((OUT / "RAW_INPUT_LOCKS.json").read_text())
    target_lock = raw_locks["replogle_rpe1_target"]
    if (
        str(TARGET) != str(target_lock["path"])
        or TARGET.stat().st_size != int(target_lock["bytes"])
        or sha256_file(TARGET) != str(target_lock["sha256"])
    ):
        raise RuntimeError("frozen RPE1 raw file changed")
    panel = pd.read_csv(ASSETS / "GENE_PANEL.csv", keep_default_na=False).sort_values(
        "panel_index"
    )
    query = pd.read_csv(OUT / "E192_QUERY_MANIFEST.csv", keep_default_na=False)
    assignments = pd.read_csv(
        OUT / "E192_TARGET_CELL_TASK_ASSIGNMENTS.csv", keep_default_na=False
    )
    order = pd.read_csv(
        OUT / "pretruth_release/tables/QUERY_ORDER.csv", keep_default_na=False
    )
    if (
        len(panel) != N_GENES
        or len(query) != N_QUERIES
        or len(assignments) != N_TARGET_ROWS
        or order.task_id.astype(str).tolist() != query.task_id.astype(str).tolist()
    ):
        raise RuntimeError("E192 target truth manifest/order contract failed")
    controls = load_npz(ASSETS / "TARGET_CONTROL_PROFILES.npz")
    target = ad.read_h5ad(TARGET, backed="r")
    try:
        target_var = {
            str(gene): index for index, gene in enumerate(target.var_names.astype(str))
        }
        columns = np.asarray(
            [target_var[str(gene)] for gene in panel.gene_name], dtype=np.int64
        )
        rows = np.sort(assignments.target_row_index.astype(int).unique())
        if len(rows) != N_TARGET_ROWS:
            raise RuntimeError("E192 target perturbation row count changed")
        builder = import_asset_builder()
        matrix = builder.import_e190_helper().read_rows_cols(target, rows, columns)
        row_position = {int(row): index for index, row in enumerate(rows)}
        true_effects: dict[str, np.ndarray] = {}
        truth_index: list[dict[str, object]] = []
        for task_id, group in assignments.groupby("task_id", observed=True, sort=False):
            positions = [
                row_position[int(row)] for row in group.target_row_index
            ]
            batch = str(group.batch.iloc[0])
            gene = str(group.gene.iloc[0])
            if (
                len(set(group.batch.astype(str))) != 1
                or len(set(group.gene.astype(str))) != 1
            ):
                raise RuntimeError(f"mixed metadata inside E192 task {task_id}")
            effect = (
                matrix[positions].mean(axis=0) - controls[f"TARGET::{batch}"]
            ).astype(np.float32)
            true_effects[str(task_id)] = effect
            truth_index.append(
                {
                    "task_id": str(task_id),
                    "batch": batch,
                    "gene": gene,
                    "n_target_cells": len(group),
                }
            )
        if set(true_effects) != set(query.task_id.astype(str)):
            raise RuntimeError("E192 target truth IDs differ from frozen query IDs")
        ordered = {
            task_id: true_effects[task_id] for task_id in query.task_id.astype(str)
        }
        (TRUTH / "arrays").mkdir(parents=True)
        (TRUTH / "tables").mkdir()
        with (TRUTH / "arrays/TARGET_TRUE_EFFECTS.npz").open("wb") as handle:
            np.savez_compressed(handle, **ordered)
        pd.DataFrame(truth_index).set_index("task_id").loc[
            query.task_id.astype(str)
        ].reset_index().to_csv(TRUTH / "tables/TARGET_TRUTH_INDEX.csv", index=False)
        scale = {
            "matrix": "target_perturbation",
            "rows": matrix.shape[0],
            "columns": matrix.shape[1],
            "minimum": float(matrix.min()),
            "maximum": float(matrix.max()),
            "mean": float(matrix.mean()),
            "median": float(np.median(matrix)),
            "fraction_zero": float(np.mean(matrix == 0)),
            "all_finite": bool(np.isfinite(matrix).all()),
            "all_nonnegative": bool(np.all(matrix >= 0)),
            "normalization": "per-cell full-library scale to 10000, then log1p",
        }
        pd.DataFrame([scale]).to_csv(
            TRUTH / "tables/TARGET_INPUT_SCALE_AUDIT.csv", index=False
        )
        status = {
            "experiment": "E192",
            "stage": "TARGET_TRUTH_BUILD",
            "status": "PASS",
            "n_target_tasks": len(ordered),
            "target_perturbation_x_rows_read": len(rows),
            "normalization": scale["normalization"],
            **freeze,
        }
        (TRUTH / "TARGET_TRUTH_BUILD_STATUS.json").write_text(
            json.dumps(status, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        files = sorted(path for path in TRUTH.rglob("*") if path.is_file())
        pd.DataFrame(
            [
                {
                    "path": path.relative_to(TRUTH).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in files
            ]
        ).to_csv(TRUTH / "TRUTH_LOCKS.csv", index=False)
        print(json.dumps(status, ensure_ascii=False, indent=2))
    finally:
        target.file.close()


if __name__ == "__main__":
    main()
