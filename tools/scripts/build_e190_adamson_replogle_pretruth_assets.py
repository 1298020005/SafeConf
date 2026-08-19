#!/usr/bin/env python3
"""Build E190 model assets without reading Replogle perturbation expression."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
OUT = ROOT / "docs/实验结果/E190_adamson_to_replogle_direct_transfer_20260729"
DATA = Path("/home/yyf/data/safeconf_e190_adamson_replogle")
ASSETS = DATA / "model_assets"
SOURCE = Path(
    "/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/"
    "AdamsonWeissman2016_GSM2406681_10X010.h5ad"
)
TARGET = Path(
    "/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/"
    "ReplogleWeissman2022_K562_essential.h5ad"
)
SCGPT_CHECKPOINT = Path(
    "/home/yyf/archive/code/20260519_0958_home_cleanup/moved_top_level/"
    "codex_scgpt_attnres_workspace/checkpoints/whole-human"
)
GO_SOURCE = Path("/home/yyf/data/gears_formal_baselines_v2/frangieh_local_atlas/go.csv")
N_GENES = 512
TOP_COEXPRESSION = 20
SCGPT_SYMBOL_ALIASES = {
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
        raise RuntimeError("E190 asset build requires a named branch")
    remote_heads = {}
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
        OUT / "PREREG_ANALYSIS_PLAN.md",
        OUT / "METADATA_FREEZE_STATUS.json",
        OUT / "E190_SELECTED_GENES.csv",
        OUT / "E190_QUERY_MANIFEST.csv",
        OUT / "E190_SOURCE_CELL_FOLD_ASSIGNMENTS.csv",
        OUT / "E190_TARGET_CELL_TASK_ASSIGNMENTS.csv",
        OUT / "E190_TARGET_CONTROL_ASSIGNMENTS.csv",
    ):
        require_committed(path, head)
    return {"git_head": head, "git_branch": branch, "remote_heads": remote_heads}


def read_rows_cols(
    adata: Any,
    row_indices: np.ndarray,
    column_indices: np.ndarray,
    chunk_size: int = 512,
) -> np.ndarray:
    rows = np.asarray(row_indices, dtype=np.int64)
    columns = np.asarray(column_indices, dtype=np.int64)
    if len(np.unique(rows)) != len(rows) or np.any(np.diff(rows) < 0):
        raise RuntimeError("expression row requests must be unique and sorted")
    blocks = []
    for start in range(0, len(rows), chunk_size):
        take = rows[start : start + chunk_size]
        block = adata.X[take]
        if sparse.issparse(block):
            library_size = np.asarray(block.sum(axis=1)).ravel().astype(np.float64)
            selected = block[:, columns].toarray()
        else:
            dense = np.asarray(block)
            library_size = dense.sum(axis=1, dtype=np.float64)
            selected = dense[:, columns]
        if not np.isfinite(library_size).all() or np.any(library_size <= 0):
            raise RuntimeError("invalid full-library size during E190 normalization")
        selected = np.log1p(
            np.asarray(selected, dtype=np.float64)
            / library_size[:, None]
            * 10_000.0
        ).astype(np.float32)
        if selected.shape != (len(take), len(columns)) or not np.isfinite(selected).all():
            raise RuntimeError("invalid expression block")
        blocks.append(selected)
    return np.concatenate(blocks, axis=0)


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(
            handle, **{str(key): np.asarray(value, np.float32) for key, value in arrays.items()}
        )


def scale_audit(name: str, matrix: np.ndarray) -> dict[str, Any]:
    values = np.asarray(matrix, float)
    return {
        "matrix": name,
        "rows": values.shape[0],
        "columns": values.shape[1],
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "fraction_zero": float(np.mean(values == 0)),
        "all_finite": bool(np.isfinite(values).all()),
        "all_nonnegative": bool(np.all(values >= 0)),
    }


def coexpression_edges(matrix: np.ndarray, genes: list[str]) -> pd.DataFrame:
    correlations = np.corrcoef(np.asarray(matrix, float), rowvar=False)
    correlations = np.nan_to_num(correlations, nan=0.0, posinf=0.0, neginf=0.0)
    absolute = np.abs(correlations)
    rows = []
    for target_index, target in enumerate(genes):
        order = np.argsort(-absolute[:, target_index], kind="stable")
        kept = [index for index in order if index != target_index][:TOP_COEXPRESSION]
        rows.extend(
            {
                "source": genes[source_index],
                "target": target,
                "importance": float(absolute[source_index, target_index]),
            }
            for source_index in kept
        )
    return pd.DataFrame(rows)


def main() -> None:
    freeze = verify_freeze()
    selected_genes = pd.read_csv(OUT / "E190_SELECTED_GENES.csv", keep_default_na=False)
    selected_gene_set = set(selected_genes.gene.astype(str))
    query = pd.read_csv(OUT / "E190_QUERY_MANIFEST.csv", keep_default_na=False)
    source_assignments = pd.read_csv(
        OUT / "E190_SOURCE_CELL_FOLD_ASSIGNMENTS.csv", keep_default_na=False
    )
    target_control_assignments = pd.read_csv(
        OUT / "E190_TARGET_CONTROL_ASSIGNMENTS.csv", keep_default_na=False
    )

    source = ad.read_h5ad(SOURCE, backed="r")
    target = ad.read_h5ad(TARGET, backed="r")
    try:
        source_var = {str(gene): index for index, gene in enumerate(source.var_names.astype(str))}
        target_var = {str(gene): index for index, gene in enumerate(target.var_names.astype(str))}
        vocab = set(json.loads((SCGPT_CHECKPOINT / "vocab.json").read_text()))
        expression_common = set(source_var) & set(target_var)
        common = sorted(
            gene
            for gene in expression_common
            if SCGPT_SYMBOL_ALIASES.get(gene, gene) in vocab
        )
        if not selected_gene_set.issubset(common):
            raise RuntimeError("a frozen E190 perturbation gene is absent from common scGPT axis")

        source_control_rows = np.flatnonzero(
            source.obs.nperts.astype(str).eq("0").to_numpy()
        )
        candidate_columns = np.asarray([source_var[gene] for gene in common], dtype=np.int64)
        source_control_candidates = read_rows_cols(
            source, source_control_rows, candidate_columns
        )
        candidate_means = source_control_candidates.mean(axis=0)
        candidate_mean = dict(zip(common, candidate_means))
        ranked_remaining = sorted(
            set(common) - selected_gene_set,
            key=lambda gene: (-float(candidate_mean[gene]), gene),
        )
        panel_genes = sorted(selected_gene_set)
        used_tokens = {
            SCGPT_SYMBOL_ALIASES.get(gene, gene) for gene in panel_genes
        }
        for gene in ranked_remaining:
            token = SCGPT_SYMBOL_ALIASES.get(gene, gene)
            if token in used_tokens:
                continue
            panel_genes.append(gene)
            used_tokens.add(token)
            if len(panel_genes) == N_GENES:
                break
        panel_tokens = [SCGPT_SYMBOL_ALIASES.get(gene, gene) for gene in panel_genes]
        if (
            len(panel_genes) != N_GENES
            or len(set(panel_genes)) != N_GENES
            or len(set(panel_tokens)) != N_GENES
        ):
            raise RuntimeError("E190 common panel construction failed")
        common_position = {gene: index for index, gene in enumerate(common)}
        source_control_matrix = source_control_candidates[
            :, [common_position[gene] for gene in panel_genes]
        ]
        source_control_profile = source_control_matrix.mean(axis=0).astype(np.float32)

        target_control_rows = np.sort(
            target_control_assignments.target_row_index.astype(int).unique()
        )
        target_panel_columns = np.asarray(
            [target_var[gene] for gene in panel_genes], dtype=np.int64
        )
        target_control_matrix = read_rows_cols(
            target, target_control_rows, target_panel_columns
        )
        target_row_position = {
            int(row): index for index, row in enumerate(target_control_rows)
        }
        target_controls: dict[str, np.ndarray] = {}
        for batch, group in target_control_assignments.groupby(
            "batch", observed=True, sort=True
        ):
            positions = [
                target_row_position[int(row)] for row in group.target_row_index
            ]
            target_controls[f"TARGET::{batch}"] = target_control_matrix[positions].mean(
                axis=0
            )

        source_target_rows = np.sort(
            source_assignments.source_row_index.astype(int).unique()
        )
        source_panel_columns = np.asarray(
            [source_var[gene] for gene in panel_genes], dtype=np.int64
        )
        source_target_matrix = read_rows_cols(
            source, source_target_rows, source_panel_columns
        )
        source_row_position = {
            int(row): index for index, row in enumerate(source_target_rows)
        }

        train_effects: dict[str, np.ndarray] = {}
        validation_effects: dict[str, np.ndarray] = {}
        train_rows = []
        validation_rows = []
        for (gene, guide, fold), group in source_assignments.groupby(
            ["gene", "perturbation", "fold"], observed=True, sort=True
        ):
            positions = [
                source_row_position[int(row)] for row in group.source_row_index
            ]
            effect = (
                source_target_matrix[positions].mean(axis=0) - source_control_profile
            ).astype(np.float32)
            task_id = f"ADAM::{gene}::{guide}::fold{int(fold)}"
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
        if len(train_effects) != 216 or len(validation_effects) != 54:
            raise RuntimeError("E190 source pseudobulk task counts changed")

        source_gene_effects = {}
        for gene, group in source_assignments.groupby("gene", observed=True, sort=True):
            positions = [
                source_row_position[int(row)] for row in group.source_row_index
            ]
            source_gene_effects[str(gene)] = (
                source_target_matrix[positions].mean(axis=0) - source_control_profile
            ).astype(np.float32)

        query = query.copy()
        query["context_key"] = "TARGET::" + query.batch.astype(str)
        query["gene"] = query.gene.astype(str)
        if not set(query.context_key).issubset(target_controls):
            raise RuntimeError("target query lacks a control profile")

        if ASSETS.exists():
            shutil.rmtree(ASSETS)
        ASSETS.mkdir(parents=True)
        panel = pd.DataFrame(
            {
                "panel_index": range(N_GENES),
                "gene_name": panel_genes,
                "scgpt_token": panel_tokens,
                "source_column_index": [source_var[gene] for gene in panel_genes],
                "target_column_index": [target_var[gene] for gene in panel_genes],
                "panel_role": [
                    "TRANSFER_TARGET" if gene in selected_gene_set else "SOURCE_CONTROL_HIGH_EXPRESSION"
                    for gene in panel_genes
                ],
                "source_control_mean_expression": [
                    float(candidate_mean[gene]) for gene in panel_genes
                ],
            }
        )
        panel.to_csv(ASSETS / "GENE_PANEL.csv", index=False)
        pd.DataFrame(train_rows).to_csv(ASSETS / "TRAIN_TASKS.csv", index=False)
        pd.DataFrame(validation_rows).to_csv(ASSETS / "VALIDATION_TASKS.csv", index=False)
        query.to_csv(ASSETS / "QUERY_TASKS.csv", index=False)
        save_npz(ASSETS / "TRAIN_EFFECTS.npz", train_effects)
        save_npz(ASSETS / "VALIDATION_EFFECTS.npz", validation_effects)
        save_npz(
            ASSETS / "SOURCE_CONTROL_PROFILE.npz",
            {"SOURCE::K562": source_control_profile},
        )
        save_npz(ASSETS / "TARGET_CONTROL_PROFILES.npz", target_controls)
        save_npz(ASSETS / "SOURCE_GENE_EFFECTS.npz", source_gene_effects)
        coexpression_edges(source_control_matrix, panel_tokens).to_csv(
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
            raise RuntimeError("E190 frozen GO subgraph is empty")
        go.to_csv(ASSETS / "GO_EDGES_PANEL.csv", index=False)

        scale = pd.DataFrame(
            [
                scale_audit("source_control", source_control_matrix),
                scale_audit("source_target", source_target_matrix),
                scale_audit("target_control", target_control_matrix),
            ]
        )
        scale.to_csv(OUT / "PRETRUTH_INPUT_SCALE_AUDIT.csv", index=False)
        source_lock = {
            "path": str(SOURCE),
            "bytes": SOURCE.stat().st_size,
            "sha256": sha256_file(SOURCE),
        }
        target_lock = {
            "path": str(TARGET),
            "bytes": TARGET.stat().st_size,
            "sha256": sha256_file(TARGET),
        }
        (OUT / "RAW_SOURCE_LOCKS.json").write_text(
            json.dumps(
                {"adamson_source": source_lock, "replogle_target": target_lock},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
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
            "experiment": "E190",
            "stage": "PRETRUTH_MODEL_ASSET_BUILD",
            "status": "PASS",
            "n_panel_genes": N_GENES,
            "n_transfer_genes": len(selected_gene_set),
            "n_train_tasks": len(train_effects),
            "n_validation_tasks": len(validation_effects),
            "n_target_queries": len(query),
            "n_target_control_profiles": len(target_controls),
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
