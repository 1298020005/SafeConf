#!/usr/bin/env python3
"""Freeze E192 tasks from metadata only; never read target expression."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import anndata as ad
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
OUT = ROOT / "docs/实验结果/E192_adamson_to_replogle_rpe1_locked_transfer_20260729"
SOURCE_SELECTION = (
    ROOT
    / "docs/实验结果/E190_adamson_to_replogle_direct_transfer_20260729"
    / "E190_SELECTED_GENES.csv"
)
SOURCE_ASSIGNMENTS = (
    ROOT
    / "docs/实验结果/E190_adamson_to_replogle_direct_transfer_20260729"
    / "E190_SOURCE_CELL_FOLD_ASSIGNMENTS.csv"
)
TARGET = Path(
    "/home/yyf/data/singlecell_perturbation_atlas/official_generalization/"
    "Replogle_RPE1essential.h5ad"
)
MIN_TARGET_CELLS = 5
MIN_TARGET_CONTROLS = 20
EXPECTED_GENES = 21
EXPECTED_TASKS = 175
EXPECTED_TARGET_ROWS = 1086
EXPECTED_BATCHES = 53
EXPECTED_SOURCE_TRAIN_TASKS = 104
EXPECTED_SOURCE_VALIDATION_TASKS = 26


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


def verify_freeze() -> dict[str, object]:
    head = git_text("rev-parse", "HEAD")
    branch = git_text("branch", "--show-current")
    if not branch:
        raise RuntimeError("E192 metadata freeze requires a named branch")
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
    for path in (
        RUNNER,
        OUT / "PREREG_CONFIRMATION_PLAN.md",
        SOURCE_SELECTION,
        SOURCE_ASSIGNMENTS,
    ):
        require_committed(path, head)
    return {
        "git_head": head,
        "git_branch": branch,
        "remote_heads": remote_heads,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    freeze = verify_freeze()
    fixed_genes = set(
        pd.read_csv(SOURCE_SELECTION, keep_default_na=False).gene.astype(str)
    )
    source = pd.read_csv(SOURCE_ASSIGNMENTS, keep_default_na=False)
    target = ad.read_h5ad(TARGET, backed="r")
    try:
        obs = target.obs.copy()
        obs.insert(0, "target_cell_id", obs.index.astype(str))
        obs.insert(0, "target_row_index", range(len(obs)))
        obs["batch"] = obs.gem_group.astype(str)
        obs["gene"] = obs.gene.astype(str)
        is_control = obs.transcript.astype(str).eq("non-targeting")

        controls = obs.loc[
            is_control,
            ["target_row_index", "target_cell_id", "batch", "gene", "transcript"],
        ].copy()
        control_counts = controls.groupby("batch", observed=True).size()
        eligible_control_batches = set(
            control_counts.loc[control_counts >= MIN_TARGET_CONTROLS].index.astype(str)
        )
        perturb = obs.loc[
            ~is_control & obs.gene.isin(fixed_genes),
            ["target_row_index", "target_cell_id", "batch", "gene", "transcript"],
        ].copy()
        task_counts = (
            perturb.groupby(["batch", "gene"], observed=True)
            .size()
            .rename("n_target_cells")
            .reset_index()
        )
        task_counts = task_counts.loc[
            task_counts.batch.isin(eligible_control_batches)
            & task_counts.n_target_cells.ge(MIN_TARGET_CELLS)
        ].copy()
        eligible_genes = sorted(task_counts.gene.unique())
        task_counts = task_counts.loc[task_counts.gene.isin(eligible_genes)].copy()
        task_counts["task_id"] = (
            "RPE1::" + task_counts.batch + "::" + task_counts.gene
        )
        task_counts = task_counts.sort_values(["gene", "batch"]).reset_index(drop=True)
        eligible_ids = set(task_counts.task_id)
        perturb["task_id"] = "RPE1::" + perturb.batch + "::" + perturb.gene
        target_assignments = perturb.loc[perturb.task_id.isin(eligible_ids)].copy()
        query_batches = set(task_counts.batch)
        controls = controls.loc[controls.batch.isin(query_batches)].copy()

        source = source.loc[source.gene.astype(str).isin(eligible_genes)].copy()
        source_train_tasks = source.loc[source.fold.astype(int).ne(4)].groupby(
            ["gene", "perturbation", "fold"], observed=True
        ).ngroups
        source_validation_tasks = source.loc[source.fold.astype(int).eq(4)].groupby(
            ["gene", "perturbation", "fold"], observed=True
        ).ngroups
        observed_counts = target_assignments.groupby("task_id", observed=True).size()
        expected_counts = task_counts.set_index("task_id").n_target_cells.astype(int)
        checks = {
            "genes": len(eligible_genes),
            "tasks": len(task_counts),
            "target_rows": len(target_assignments),
            "batches": task_counts.batch.nunique(),
            "source_train_tasks": source_train_tasks,
            "source_validation_tasks": source_validation_tasks,
        }
        expected = {
            "genes": EXPECTED_GENES,
            "tasks": EXPECTED_TASKS,
            "target_rows": EXPECTED_TARGET_ROWS,
            "batches": EXPECTED_BATCHES,
            "source_train_tasks": EXPECTED_SOURCE_TRAIN_TASKS,
            "source_validation_tasks": EXPECTED_SOURCE_VALIDATION_TASKS,
        }
        if checks != expected:
            raise RuntimeError(f"E192 metadata counts changed: {checks} != {expected}")
        if not observed_counts.sort_index().equals(expected_counts.sort_index()):
            raise RuntimeError("E192 target cell assignments do not match task counts")
        if set(source.gene.astype(str)) != set(eligible_genes):
            raise RuntimeError("E192 source assignments missing an eligible target gene")

        pd.DataFrame(
            {
                "gene": eligible_genes,
                "selection_rule": (
                    "E190-frozen gene; RPE1 metadata >=5 cells in >=1 batch"
                ),
                "selection_sha256": [
                    hashlib.sha256(f"E192\\0{gene}".encode()).hexdigest()
                    for gene in eligible_genes
                ],
            }
        ).to_csv(OUT / "E192_SELECTED_GENES.csv", index=False)
        task_counts.to_csv(OUT / "E192_QUERY_MANIFEST.csv", index=False)
        source.to_csv(OUT / "E192_SOURCE_CELL_FOLD_ASSIGNMENTS.csv", index=False)
        target_assignments.to_csv(
            OUT / "E192_TARGET_CELL_TASK_ASSIGNMENTS.csv", index=False
        )
        controls.to_csv(OUT / "E192_TARGET_CONTROL_ASSIGNMENTS.csv", index=False)
        status = {
            "experiment": "E192",
            "stage": "METADATA_ONLY_FREEZE",
            "status": "PASS",
            "source_gene_candidates": len(fixed_genes),
            "n_selected_genes": len(eligible_genes),
            "n_target_tasks": len(task_counts),
            "n_target_selected_cells": len(target_assignments),
            "n_target_control_cells": len(controls),
            "n_target_batches": task_counts.batch.nunique(),
            "n_source_train_tasks": source_train_tasks,
            "n_source_validation_tasks": source_validation_tasks,
            "target_file": str(TARGET),
            "target_bytes": TARGET.stat().st_size,
            "target_sha256": sha256_file(TARGET),
            "source_x_values_read": 0,
            "target_x_values_read": 0,
            "selection_used_expression_values": False,
            **freeze,
        }
        (OUT / "METADATA_FREEZE_STATUS.json").write_text(
            json.dumps(status, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(status, ensure_ascii=False, indent=2))
    finally:
        target.file.close()


if __name__ == "__main__":
    main()
