#!/usr/bin/env python3
"""Build E192 assets without reading RPE1 perturbation expression."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
OUT = ROOT / "docs/实验结果/E192_adamson_to_replogle_rpe1_locked_transfer_20260729"
DATA = Path("/home/yyf/data/safeconf_e192_adamson_rpe1")
ASSETS = DATA / "model_assets"
SOURCE = Path(
    "/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/"
    "AdamsonWeissman2016_GSM2406681_10X010.h5ad"
)
TARGET = Path(
    "/home/yyf/data/singlecell_perturbation_atlas/official_generalization/"
    "Replogle_RPE1essential.h5ad"
)
SCGPT_CHECKPOINT = Path(
    "/home/yyf/archive/code/20260519_0958_home_cleanup/moved_top_level/"
    "codex_scgpt_attnres_workspace/checkpoints/whole-human"
)
GO_SOURCE = Path("/home/yyf/data/gears_formal_baselines_v2/frangieh_local_atlas/go.csv")
E190_HELPER_PATH = ROOT / "tools/scripts/build_e190_adamson_replogle_pretruth_assets.py"
E190_RAW_LOCKS = (
    ROOT
    / "docs/实验结果/E190_adamson_to_replogle_direct_transfer_20260729"
    / "RAW_SOURCE_LOCKS.json"
)
N_GENES = 512
N_TRANSFER_GENES = 21
N_TRAIN_TASKS = 104
N_VALIDATION_TASKS = 26
N_QUERIES = 175
N_TARGET_CONTROLS = 1905
N_TARGET_CONTROL_PROFILES = 53
ALIASES = {
    "DARS": "DARS1",
    "HARS": "HARS1",
    "MARS": "MARS1",
    "QARS": "QARS1",
    "TARS": "TARS1",
}


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


def verify_freeze() -> dict[str, Any]:
    head = git_text("rev-parse", "HEAD")
    branch = git_text("branch", "--show-current")
    if not branch:
        raise RuntimeError("E192 asset build requires a named branch")
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
        OUT / "METADATA_FREEZE_STATUS.json",
        OUT / "E192_SELECTED_GENES.csv",
        OUT / "E192_QUERY_MANIFEST.csv",
        OUT / "E192_SOURCE_CELL_FOLD_ASSIGNMENTS.csv",
        OUT / "E192_TARGET_CELL_TASK_ASSIGNMENTS.csv",
        OUT / "E192_TARGET_CONTROL_ASSIGNMENTS.csv",
    ):
        require_committed(path, head)
    return {"git_head": head, "git_branch": branch, "remote_heads": remote_heads}


def import_e190_helper() -> Any:
    spec = importlib.util.spec_from_file_location("e192_e190_asset_helper", E190_HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen E190 normalization helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            **{str(key): np.asarray(value, np.float32) for key, value in arrays.items()},
        )


def main() -> None:
    freeze = verify_freeze()
    metadata = json.loads((OUT / "METADATA_FREEZE_STATUS.json").read_text())
    if (
        metadata.get("status") != "PASS"
        or metadata.get("target_x_values_read") != 0
        or metadata.get("n_target_tasks") != N_QUERIES
        or metadata.get("n_selected_genes") != N_TRANSFER_GENES
    ):
        raise RuntimeError("E192 metadata freeze does not authorize asset build")
    if (
        TARGET.stat().st_size != int(metadata["target_bytes"])
        or sha256_file(TARGET) != str(metadata["target_sha256"])
    ):
        raise RuntimeError("E192 target raw file differs from metadata freeze")

    helper = import_e190_helper()
    selected = pd.read_csv(OUT / "E192_SELECTED_GENES.csv", keep_default_na=False)
    selected_genes = set(selected.gene.astype(str))
    query = pd.read_csv(OUT / "E192_QUERY_MANIFEST.csv", keep_default_na=False)
    source_assignments = pd.read_csv(
        OUT / "E192_SOURCE_CELL_FOLD_ASSIGNMENTS.csv", keep_default_na=False
    )
    target_controls = pd.read_csv(
        OUT / "E192_TARGET_CONTROL_ASSIGNMENTS.csv", keep_default_na=False
    )
    source = ad.read_h5ad(SOURCE, backed="r")
    target = ad.read_h5ad(TARGET, backed="r")
    try:
        source_var = {str(g): i for i, g in enumerate(source.var_names.astype(str))}
        target_var = {str(g): i for i, g in enumerate(target.var_names.astype(str))}
        vocab = set(json.loads((SCGPT_CHECKPOINT / "vocab.json").read_text()))
        common = sorted(
            gene
            for gene in set(source_var) & set(target_var)
            if ALIASES.get(gene, gene) in vocab
        )
        if not selected_genes.issubset(common):
            raise RuntimeError("an E192 transfer gene is absent from the common scGPT axis")

        source_control_rows = np.flatnonzero(
            source.obs.nperts.astype(str).eq("0").to_numpy()
        )
        common_columns = np.asarray([source_var[g] for g in common], dtype=np.int64)
        source_control_candidates = helper.read_rows_cols(
            source, source_control_rows, common_columns
        )
        common_means = source_control_candidates.mean(axis=0)
        common_mean = dict(zip(common, common_means))
        panel_genes = sorted(selected_genes)
        used_tokens = {ALIASES.get(g, g) for g in panel_genes}
        remaining = sorted(
            set(common) - selected_genes,
            key=lambda gene: (-float(common_mean[gene]), gene),
        )
        for gene in remaining:
            token = ALIASES.get(gene, gene)
            if token in used_tokens:
                continue
            panel_genes.append(gene)
            used_tokens.add(token)
            if len(panel_genes) == N_GENES:
                break
        panel_tokens = [ALIASES.get(g, g) for g in panel_genes]
        if len(panel_genes) != N_GENES or len(set(panel_tokens)) != N_GENES:
            raise RuntimeError("E192 common 512-gene panel construction failed")
        common_position = {gene: i for i, gene in enumerate(common)}
        source_control_matrix = source_control_candidates[
            :, [common_position[gene] for gene in panel_genes]
        ]
        source_control_profile = source_control_matrix.mean(axis=0).astype(np.float32)

        target_control_rows = np.sort(
            target_controls.target_row_index.astype(int).unique()
        )
        if len(target_control_rows) != N_TARGET_CONTROLS:
            raise RuntimeError("E192 target control row count changed")
        target_columns = np.asarray([target_var[g] for g in panel_genes], dtype=np.int64)
        target_control_matrix = helper.read_rows_cols(
            target, target_control_rows, target_columns
        )
        target_row_position = {int(row): i for i, row in enumerate(target_control_rows)}
        control_profiles: dict[str, np.ndarray] = {}
        for batch, group in target_controls.groupby("batch", observed=True, sort=True):
            positions = [
                target_row_position[int(row)] for row in group.target_row_index
            ]
            control_profiles[f"TARGET::{batch}"] = target_control_matrix[
                positions
            ].mean(axis=0)
        if len(control_profiles) != N_TARGET_CONTROL_PROFILES:
            raise RuntimeError("E192 target control profile count changed")

        source_target_rows = np.sort(
            source_assignments.source_row_index.astype(int).unique()
        )
        source_columns = np.asarray([source_var[g] for g in panel_genes], dtype=np.int64)
        source_target_matrix = helper.read_rows_cols(
            source, source_target_rows, source_columns
        )
        source_row_position = {int(row): i for i, row in enumerate(source_target_rows)}
        train_effects: dict[str, np.ndarray] = {}
        validation_effects: dict[str, np.ndarray] = {}
        train_rows: list[dict[str, object]] = []
        validation_rows: list[dict[str, object]] = []
        for (gene, guide, fold), group in source_assignments.groupby(
            ["gene", "perturbation", "fold"], observed=True, sort=True
        ):
            positions = [
                source_row_position[int(row)] for row in group.source_row_index
            ]
            effect = (
                source_target_matrix[positions].mean(axis=0) - source_control_profile
            ).astype(np.float32)
            task_id = f"ADAM192::{gene}::{guide}::fold{int(fold)}"
            row = {
                "task_id": task_id,
                "gene": str(gene),
                "guide": str(guide),
                "fold": int(fold),
                "n_cells": len(group),
                "context_key": "SOURCE::K562",
                "split": "validation" if int(fold) == 4 else "train",
            }
            if int(fold) == 4:
                validation_effects[task_id] = effect
                validation_rows.append(row)
            else:
                train_effects[task_id] = effect
                train_rows.append(row)
        if (
            len(train_effects) != N_TRAIN_TASKS
            or len(validation_effects) != N_VALIDATION_TASKS
        ):
            raise RuntimeError("E192 source pseudobulk task counts changed")

        source_gene_effects: dict[str, np.ndarray] = {}
        for gene, group in source_assignments.groupby("gene", observed=True, sort=True):
            positions = [
                source_row_position[int(row)] for row in group.source_row_index
            ]
            source_gene_effects[str(gene)] = (
                source_target_matrix[positions].mean(axis=0) - source_control_profile
            ).astype(np.float32)
        query = query.copy()
        query["context_key"] = "TARGET::" + query.batch.astype(str)
        if not set(query.context_key).issubset(control_profiles):
            raise RuntimeError("E192 query lacks a target control profile")

        if ASSETS.exists():
            shutil.rmtree(ASSETS)
        ASSETS.mkdir(parents=True)
        panel = pd.DataFrame(
            {
                "panel_index": range(N_GENES),
                "gene_name": panel_genes,
                "scgpt_token": panel_tokens,
                "source_column_index": [source_var[g] for g in panel_genes],
                "target_column_index": [target_var[g] for g in panel_genes],
                "panel_role": [
                    "TRANSFER_TARGET"
                    if gene in selected_genes
                    else "SOURCE_CONTROL_HIGH_EXPRESSION"
                    for gene in panel_genes
                ],
                "source_control_mean_expression": [
                    float(common_mean[g]) for g in panel_genes
                ],
            }
        )
        panel.to_csv(ASSETS / "GENE_PANEL.csv", index=False)
        pd.DataFrame(train_rows).to_csv(ASSETS / "TRAIN_TASKS.csv", index=False)
        pd.DataFrame(validation_rows).to_csv(
            ASSETS / "VALIDATION_TASKS.csv", index=False
        )
        query.to_csv(ASSETS / "QUERY_TASKS.csv", index=False)
        save_npz(ASSETS / "TRAIN_EFFECTS.npz", train_effects)
        save_npz(ASSETS / "VALIDATION_EFFECTS.npz", validation_effects)
        save_npz(
            ASSETS / "SOURCE_CONTROL_PROFILE.npz",
            {"SOURCE::K562": source_control_profile},
        )
        save_npz(ASSETS / "TARGET_CONTROL_PROFILES.npz", control_profiles)
        save_npz(ASSETS / "SOURCE_GENE_EFFECTS.npz", source_gene_effects)
        helper.coexpression_edges(source_control_matrix, panel_tokens).to_csv(
            ASSETS / "SOURCE_CONTROL_COEXPRESSION_EDGES.csv", index=False
        )
        go = pd.read_csv(GO_SOURCE)
        go = go.loc[
            go.source.astype(str).isin(panel_tokens)
            & go.target.astype(str).isin(panel_tokens)
        ].copy()
        go = (
            go.sort_values(["target", "importance"], ascending=[True, False])
            .groupby("target", as_index=False, group_keys=False)
            .head(21)
        )
        if go.empty:
            raise RuntimeError("E192 frozen GO subgraph is empty")
        go.to_csv(ASSETS / "GO_EDGES_PANEL.csv", index=False)

        pd.DataFrame(
            [
                helper.scale_audit("source_control", source_control_matrix),
                helper.scale_audit("source_target", source_target_matrix),
                helper.scale_audit("target_control", target_control_matrix),
            ]
        ).to_csv(OUT / "PRETRUTH_INPUT_SCALE_AUDIT.csv", index=False)
        e190_locks = json.loads(E190_RAW_LOCKS.read_text())
        source_lock = e190_locks["adamson_source"]
        if (
            str(SOURCE) != str(source_lock["path"])
            or SOURCE.stat().st_size != int(source_lock["bytes"])
            or sha256_file(SOURCE) != str(source_lock["sha256"])
        ):
            raise RuntimeError("E192 Adamson source differs from E190 frozen source")
        raw_locks = {
            "adamson_source": source_lock,
            "replogle_rpe1_target": {
                "path": str(TARGET),
                "bytes": TARGET.stat().st_size,
                "sha256": str(metadata["target_sha256"]),
            },
        }
        (OUT / "RAW_INPUT_LOCKS.json").write_text(
            json.dumps(raw_locks, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        checkpoint_locks = []
        for name in ("args.json", "vocab.json", "best_model.pt"):
            path = SCGPT_CHECKPOINT / name
            checkpoint_locks.append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        pd.DataFrame(checkpoint_locks).to_csv(
            OUT / "SCGPT_CHECKPOINT_LOCKS.csv", index=False
        )
        manifest = {
            "experiment": "E192",
            "stage": "PRETRUTH_MODEL_ASSET_BUILD",
            "status": "PASS",
            "n_panel_genes": N_GENES,
            "n_transfer_genes": len(selected_genes),
            "n_train_tasks": len(train_effects),
            "n_validation_tasks": len(validation_effects),
            "n_target_queries": len(query),
            "n_target_control_profiles": len(control_profiles),
            "target_perturbation_x_rows_read": 0,
            "target_control_x_rows_read": len(target_control_rows),
            "source_target_x_rows_read": len(source_target_rows),
            "source_control_x_rows_read": len(source_control_rows),
            "contains_target_evaluation_truth": False,
            "normalization": "per-cell full-library scale to 10000, then log1p",
            **freeze,
        }
        (ASSETS / "ASSET_MANIFEST.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        locks = []
        for path in sorted(ASSETS.iterdir()):
            if path.is_file():
                locks.append(
                    {
                        "path": path.relative_to(DATA).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        pd.DataFrame(locks).to_csv(OUT / "MODEL_ASSET_LOCKS.csv", index=False)
        (OUT / "ASSET_BUILD_STATUS.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    finally:
        source.file.close()
        target.file.close()


if __name__ == "__main__":
    main()
