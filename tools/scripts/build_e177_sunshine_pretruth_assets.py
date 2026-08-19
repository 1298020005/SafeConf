#!/usr/bin/env python3
"""Build E177 F2 assets without calibration or evaluation target truth."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "E177_sunshine_external_certificate"
EXPERIMENT_REL = Path("docs/实验结果/E177_sunshine_external_certificate_20260719")
EXPERIMENT_DIR = ROOT / EXPERIMENT_REL
FROZEN_METADATA_COMMIT = "5193056320f35e417e0d810ae863da719053f559"

SOURCE_LOCK_REL = EXPERIMENT_REL / "SOURCE_LOCK.json"
RUN_STATUS_REL = EXPERIMENT_REL / "RUN_STATUS.json"
MODEL_LOCK_REL = EXPERIMENT_REL / "MODEL_INPUT_LOCK.json"
STAT_LOCK_REL = EXPERIMENT_REL / "STATISTICAL_ANALYSIS_LOCK.json"
PLAN_REL = EXPERIMENT_REL / "PREREG_ANALYSIS_PLAN.md"
TARGETS_REL = EXPERIMENT_REL / "manifests/E177_SELECTED_TARGETS.csv"
TASKS_REL = EXPERIMENT_REL / "manifests/E177_TASK_MANIFEST.csv"
ROW_ACCESS_REL = EXPERIMENT_REL / "manifests/E177_ROW_ACCESS_MANIFEST.csv"
BUILDER_REL = Path("tools/scripts/build_e177_sunshine_pretruth_assets.py")

FROZEN_INPUTS = (
    SOURCE_LOCK_REL,
    RUN_STATUS_REL,
    MODEL_LOCK_REL,
    STAT_LOCK_REL,
    PLAN_REL,
    TARGETS_REL,
    TASKS_REL,
    ROW_ACCESS_REL,
)

DATA_ROOT = Path("/home/yyf/data/safeconf_e177_external")
ISOLATED_ROOT = DATA_ROOT / "isolated"
F2_DIR = ISOLATED_ROOT / "F2_pretruth"
PRETRUTH_PHASES = (
    "PRETRUTH_CONTROL_X",
    "PRETRUTH_TRAIN_X",
    "PRETRUTH_VALIDATION_X",
)
FORBIDDEN_PHASES = (
    "POSTGATE_CALIBRATION_TRUTH_X",
    "POSTCALIBRATION_EVALUATION_TRUTH_X",
)
TECH_GROUPS = tuple(str(value) for value in range(1, 9))
N_PANEL_GENES = 512
N_REGISTERED_TARGETS = 144
ALLOWLIST = {
    "ACCESS_ATTESTATION.json",
    "CONTROL_PROFILES.npz",
    "GENE_PANEL.csv",
    "MANIFEST.sha256",
    "PRETRUTH_TASKS.csv",
    "ROW_ACCESS_AUDIT.csv",
    "SEEN_TARGET_EFFECTS.npz",
    "TRAIN_CONTROL_COEXPRESSION_EDGES.csv",
    "TRAIN_CONTROL_PROFILE_INDEX.csv",
}


class IntegrityError(RuntimeError):
    """A frozen input or expression-access boundary did not match."""


def sha256_file(path: Path, chunk_size: int = 64 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def git_text(*args: str) -> str:
    return git(*args).stdout.decode("utf-8").strip()


def write_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def verify_dual_remote_contains_head(head: str) -> tuple[str, dict[str, str]]:
    branch = git_text("rev-parse", "--abbrev-ref", "HEAD")
    if branch == "HEAD":
        raise IntegrityError("E177 F2 requires a named Git branch")
    remote_heads: dict[str, str] = {}
    for remote in ("origin", "github"):
        fetched_ref = f"refs/remotes/{remote}/{branch}"
        result = git(
            "fetch",
            "--quiet",
            remote,
            f"refs/heads/{branch}:{fetched_ref}",
            check=False,
        )
        if result.returncode:
            raise IntegrityError(
                f"cannot verify remote {remote}: "
                f"{result.stderr.decode(errors='replace').strip()}"
            )
        remote_head = git_text("rev-parse", fetched_ref)
        if git("merge-base", "--is-ancestor", head, remote_head, check=False).returncode:
            raise IntegrityError(f"HEAD {head} is absent from {remote}/{branch}")
        remote_heads[remote] = remote_head
    return branch, remote_heads


def require_committed_file(relative: Path, head: str) -> str:
    path = ROOT / relative
    if not path.is_file():
        raise IntegrityError(f"missing required file: {relative}")
    committed = git("show", f"{head}:{relative.as_posix()}").stdout
    local = path.read_bytes()
    if committed != local:
        raise IntegrityError(f"working tree differs from HEAD for {relative}")
    return sha256_bytes(local)


def verify_frozen_state() -> dict[str, Any]:
    if git("cat-file", "-e", f"{FROZEN_METADATA_COMMIT}^{{commit}}", check=False).returncode:
        raise IntegrityError("E177 metadata-freeze commit is unavailable locally")
    head = git_text("rev-parse", "HEAD")
    if git("merge-base", "--is-ancestor", FROZEN_METADATA_COMMIT, head, check=False).returncode:
        raise IntegrityError("current HEAD does not descend from E177 metadata freeze")

    frozen_hashes: dict[str, str] = {}
    for relative in FROZEN_INPUTS:
        path = ROOT / relative
        frozen = git("show", f"{FROZEN_METADATA_COMMIT}:{relative.as_posix()}").stdout
        local = path.read_bytes()
        if frozen != local:
            raise IntegrityError(f"E177 frozen input changed after metadata freeze: {relative}")
        frozen_hashes[relative.as_posix()] = sha256_bytes(local)

    run_status = json.loads((ROOT / RUN_STATUS_REL).read_text(encoding="utf-8"))
    if run_status.get("status") != "PASS" or run_status.get("stage") != "F1_METADATA_FREEZE":
        raise IntegrityError("E177 metadata freeze is not in PASS state")
    for relative, expected in run_status.get("artifact_sha256", {}).items():
        path = EXPERIMENT_DIR / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise IntegrityError(f"E177 metadata artifact hash changed: {relative}")

    builder_sha = require_committed_file(BUILDER_REL, head)
    branch, remote_heads = verify_dual_remote_contains_head(head)
    return {
        "head": head,
        "branch": branch,
        "remote_heads": remote_heads,
        "frozen_hashes": frozen_hashes,
        "builder_sha256": builder_sha,
        "run_status": run_status,
        "source_lock": json.loads((ROOT / SOURCE_LOCK_REL).read_text(encoding="utf-8")),
        "model_lock": json.loads((ROOT / MODEL_LOCK_REL).read_text(encoding="utf-8")),
    }


def normalize_log1p(x: Any) -> sp.csr_matrix:
    matrix = x.tocsr().astype(np.float32) if sp.issparse(x) else sp.csr_matrix(np.asarray(x, dtype=np.float32))
    totals = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float32)
    if not np.isfinite(totals).all() or (totals <= 0).any():
        raise IntegrityError("invalid library size in selected X rows")
    scale = np.divide(1.0e4, totals, out=np.zeros_like(totals), where=totals > 0)
    matrix = (sp.diags(scale, dtype=np.float32) @ matrix).tocsr()
    matrix.data = np.log1p(matrix.data)
    return matrix


def read_rows_full(adata: ad.AnnData, row_indices: Sequence[int]) -> sp.csr_matrix:
    rows = np.asarray(row_indices, dtype=np.int64)
    if rows.ndim != 1 or len(rows) == 0:
        raise IntegrityError("read_rows_full requires a non-empty row index")
    if len(np.unique(rows)) != len(rows):
        raise IntegrityError("one source row requested twice in one read")
    if (rows[1:] <= rows[:-1]).any():
        raise IntegrityError("source rows must be strictly increasing")
    values = adata.X[rows, :]
    matrix = values.tocsr() if sp.issparse(values) else sp.csr_matrix(values)
    if matrix.shape != (len(rows), adata.n_vars):
        raise IntegrityError(f"unexpected X slice shape: {matrix.shape}")
    return normalize_log1p(matrix)


def save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    if not arrays:
        raise IntegrityError(f"refusing to write empty NPZ: {path}")
    ordered = {key: np.asarray(arrays[key], dtype=np.float32) for key in sorted(arrays)}
    np.savez_compressed(path, **ordered)
    with np.load(path, allow_pickle=False) as check:
        if set(check.files) != set(ordered):
            raise IntegrityError(f"NPZ keys changed after round trip: {path}")
        for key, expected in ordered.items():
            actual = check[key]
            if actual.shape != expected.shape or not np.allclose(actual, expected, atol=0, rtol=0):
                raise IntegrityError(f"NPZ value changed after round trip: {path}:{key}")


def load_manifests() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    targets = pd.read_csv(ROOT / TARGETS_REL, keep_default_na=False)
    tasks = pd.read_csv(ROOT / TASKS_REL, keep_default_na=False)
    rows = pd.read_csv(ROOT / ROW_ACCESS_REL, keep_default_na=False)
    rows["source_row_index"] = pd.to_numeric(rows.source_row_index, errors="raise").astype(int)
    rows["technical_group"] = rows.technical_group.astype(str)
    tasks["technical_group"] = tasks.technical_group.astype(str)
    if len(targets) != N_REGISTERED_TARGETS:
        raise IntegrityError("E177 selected target count changed")
    if len(tasks) != N_REGISTERED_TARGETS * len(TECH_GROUPS):
        raise IntegrityError("E177 task count changed")
    if rows.source_row_index.duplicated().any():
        raise IntegrityError("E177 row-access manifest has duplicate source rows")
    forbidden = rows.loc[rows.truth_access_phase.isin(FORBIDDEN_PHASES)]
    if forbidden.empty:
        raise IntegrityError("E177 row-access manifest lost calibration/evaluation rows")
    return targets, tasks, rows


def make_panel(
    var_names: Sequence[str],
    vocab: set[str],
    targets: pd.DataFrame,
    control_mean_all_genes: np.ndarray,
) -> pd.DataFrame:
    gene_to_index = {str(gene): index for index, gene in enumerate(var_names)}
    panel_rows: list[dict[str, Any]] = []
    used: set[str] = set()
    for target in targets.sort_values("e177_target_rank", kind="stable").itertuples(index=False):
        gene = str(target.perturbation)
        if gene not in gene_to_index or gene not in vocab:
            raise IntegrityError(f"selected target missing from expression axis or vocab: {gene}")
        used.add(gene)
        panel_rows.append(
            {
                "panel_index": len(panel_rows),
                "source_column_index": int(gene_to_index[gene]),
                "gene_name": gene,
                "scgpt_token": gene,
                "panel_role": "REGISTERED_TARGET",
                "target_split": str(target.target_split),
                "control_mean_expression": float(control_mean_all_genes[gene_to_index[gene]]),
            }
        )
    ranked = np.argsort(-np.asarray(control_mean_all_genes), kind="stable")
    for source_index in ranked.tolist():
        gene = str(var_names[int(source_index)])
        if gene in used or gene not in vocab or gene == "":
            continue
        used.add(gene)
        panel_rows.append(
            {
                "panel_index": len(panel_rows),
                "source_column_index": int(source_index),
                "gene_name": gene,
                "scgpt_token": gene,
                "panel_role": "CONTROL_HIGH_EXPRESSION",
                "target_split": "",
                "control_mean_expression": float(control_mean_all_genes[int(source_index)]),
            }
        )
        if len(panel_rows) == N_PANEL_GENES:
            break
    if len(panel_rows) != N_PANEL_GENES:
        raise IntegrityError(f"E177 panel has only {len(panel_rows)} genes")
    panel = pd.DataFrame(panel_rows)
    if panel.gene_name.nunique() != N_PANEL_GENES or panel.source_column_index.nunique() != N_PANEL_GENES:
        raise IntegrityError("E177 panel is not one-to-one")
    if int(panel.panel_role.eq("REGISTERED_TARGET").sum()) != N_REGISTERED_TARGETS:
        raise IntegrityError("registered targets are missing from E177 panel")
    return panel


def build_control_profiles(
    control_rows: pd.DataFrame,
    control_x_norm: sp.csr_matrix,
    panel_columns: np.ndarray,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, np.ndarray]:
    panel_matrix = control_x_norm[:, panel_columns].toarray().astype(np.float32)
    controls: dict[str, np.ndarray] = {}
    index_rows: list[dict[str, Any]] = []
    for group in TECH_GROUPS:
        mask = control_rows.technical_group.eq(group).to_numpy()
        if int(mask.sum()) < 1:
            raise IntegrityError(f"no exact controls for technical group {group}")
        controls[f"G{group}"] = panel_matrix[mask].mean(axis=0).astype(np.float32)
    for row in control_rows.itertuples(index=False):
        index_rows.append(
            {
                "source_row_index": int(row.source_row_index),
                "technical_group": str(row.technical_group),
                "perturbation": "control",
                "used_for_control_profile": True,
                "used_for_coexpression": True,
                "note": "library_sum_not_exported_after_log_normalization",
            }
        )
    return controls, pd.DataFrame(index_rows), panel_matrix


def build_coexpression(panel_matrix: np.ndarray, panel: pd.DataFrame) -> pd.DataFrame:
    values = np.asarray(panel_matrix, dtype=np.float32)
    if values.shape[1] != N_PANEL_GENES:
        raise IntegrityError("control panel matrix shape changed")
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(values, rowvar=False)
    corr = np.nan_to_num(np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0)
    genes = panel.scgpt_token.astype(str).tolist()
    rows: list[dict[str, Any]] = []
    for target_i, target in enumerate(genes):
        candidates = np.argsort(corr[:, target_i])[-11:][::-1].tolist()
        if target_i not in candidates:
            candidates.append(target_i)
        for source_i in candidates:
            importance = float(corr[int(source_i), target_i])
            if importance >= 0.4 or int(source_i) == target_i:
                rows.append(
                    {
                        "source": genes[int(source_i)],
                        "target": target,
                        "importance": importance,
                        "source_panel_index": int(source_i),
                        "target_panel_index": int(target_i),
                        "rule": "absolute_control_correlation_top10_threshold_0.4_self_included",
                    }
                )
    edges = pd.DataFrame(rows)
    if edges.empty or set(genes) - set(edges.target.astype(str)):
        raise IntegrityError("coexpression graph is missing panel genes")
    return edges


def aggregate_target_effects(
    adata: ad.AnnData,
    rows: pd.DataFrame,
    panel_columns: np.ndarray,
    controls: Mapping[str, np.ndarray],
    batch_size: int,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, list[dict[str, Any]]]:
    allowed = rows.loc[rows.truth_access_phase.isin(("PRETRUTH_TRAIN_X", "PRETRUTH_VALIDATION_X"))].copy()
    allowed = allowed.sort_values("source_row_index", kind="stable").reset_index(drop=True)
    task_sum: defaultdict[str, np.ndarray] = defaultdict(lambda: np.zeros(len(panel_columns), dtype=np.float64))
    task_count: defaultdict[str, int] = defaultdict(int)
    task_meta: dict[str, dict[str, Any]] = {}
    access_rows: list[dict[str, Any]] = []

    for start in range(0, len(allowed), batch_size):
        batch = allowed.iloc[start : start + batch_size].copy()
        norm = read_rows_full(adata, batch.source_row_index.astype(int).tolist())
        panel_values = norm[:, panel_columns].toarray().astype(np.float32)
        if len(batch) != len(panel_values):
            raise IntegrityError("target batch metadata and X rows have different lengths")
        for meta, vector in zip(batch.itertuples(index=False), panel_values):
            group = str(meta.technical_group)
            task_id = f"E177::G{group}::{meta.perturbation}"
            key = f"G{group}"
            if key not in controls:
                raise IntegrityError(f"missing control profile for {key}")
            task_sum[task_id] += vector
            task_count[task_id] += 1
            task_meta[task_id] = {
                "task_id": task_id,
                "technical_group": group,
                "perturbation": str(meta.perturbation),
                "target_split": str(meta.target_split),
                "truth_access_phase": str(meta.truth_access_phase),
            }
            access_rows.append(
                {
                    "source_row_index": int(meta.source_row_index),
                    "truth_access_phase": str(meta.truth_access_phase),
                    "logical_x_row_read_count": 1,
                    "asset_stage": "F2_PRETRUTH",
                    "purpose": "registered_train_or_validation_target",
                }
            )

    effects: dict[str, np.ndarray] = {}
    table_rows: list[dict[str, Any]] = []
    for task_id in sorted(task_sum):
        meta = task_meta[task_id]
        count = int(task_count[task_id])
        group_key = f"G{meta['technical_group']}"
        mean_profile = task_sum[task_id] / max(count, 1)
        effects[task_id] = (mean_profile - controls[group_key]).astype(np.float32)
        table_rows.append({**meta, "n_exact_cells_merged": count, "effect_asset_key": task_id})
    return effects, pd.DataFrame(table_rows), access_rows


def write_manifest(directory: Path) -> str:
    hashes = {
        path.name: sha256_file(path)
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name != "MANIFEST.sha256"
    }
    write_text(directory / "MANIFEST.sha256", "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())))
    return sha256_file(directory / "MANIFEST.sha256")


def prepare_staging() -> Path:
    ISOLATED_ROOT.mkdir(parents=True, exist_ok=True)
    if F2_DIR.exists():
        raise IntegrityError(f"refusing to overwrite immutable F2 assets: {F2_DIR}")
    stale = list(ISOLATED_ROOT.glob("F2_pretruth.staging.*"))
    if stale:
        raise IntegrityError(f"staging directory already exists: {stale}")
    staging = ISOLATED_ROOT / f"F2_pretruth.staging.{os.getpid()}"
    staging.mkdir(mode=0o750, parents=True)
    return staging


def build(batch_size: int) -> dict[str, Any]:
    frozen = verify_frozen_state()
    targets, tasks, rows = load_manifests()
    source_lock = frozen["source_lock"]
    source = Path(source_lock["source_path"])
    if not source.is_file() or sha256_file(source) != source_lock["source_sha256"]:
        raise IntegrityError("E177 source bytes changed after metadata freeze")
    vocab = set(json.loads(Path(frozen["model_lock"]["scgpt_vocab"]["path"]).read_text()))

    staging = prepare_staging()
    try:
        adata = ad.read_h5ad(source, backed="r")
        try:
            if [int(adata.n_obs), int(adata.n_vars)] != list(source_lock["source_shape"]):
                raise IntegrityError("E177 source shape changed")
            var_names = [str(value) for value in adata.var_names]
            if len(var_names) != len(set(var_names)):
                raise IntegrityError("E177 var_names are not unique")

            control_rows = rows.loc[rows.truth_access_phase.eq("PRETRUTH_CONTROL_X")].copy()
            control_rows = control_rows.sort_values("source_row_index", kind="stable").reset_index(drop=True)
            control_x_norm = read_rows_full(adata, control_rows.source_row_index.astype(int).tolist())
            control_mean_all = np.asarray(control_x_norm.mean(axis=0)).ravel()
            panel = make_panel(var_names, vocab, targets, control_mean_all)
            panel_columns = panel.source_column_index.to_numpy(dtype=np.int64)
            controls, control_profile_index, control_panel_matrix = build_control_profiles(
                control_rows, control_x_norm, panel_columns
            )
            coexpression = build_coexpression(control_panel_matrix, panel)
            effects, effect_index, target_access = aggregate_target_effects(
                adata, rows, panel_columns, controls, batch_size
            )
        finally:
            adata.file.close()

        expected_seen = int(tasks.target_split.isin(["train", "validation"]).sum())
        if len(effects) != expected_seen:
            raise IntegrityError(f"pretruth effect task count changed: {len(effects)} != {expected_seen}")
        forbidden_tasks = set(tasks.loc[tasks.target_split.isin(["calibration", "evaluation"]), "task_id"].astype(str))
        if set(effects) & forbidden_tasks:
            raise IntegrityError("calibration/evaluation target truth leaked into F2 effects")

        panel.to_csv(staging / "GENE_PANEL.csv", index=False, float_format="%.17g")
        save_npz(staging / "CONTROL_PROFILES.npz", controls)
        save_npz(staging / "SEEN_TARGET_EFFECTS.npz", effects)
        coexpression.to_csv(staging / "TRAIN_CONTROL_COEXPRESSION_EDGES.csv", index=False, float_format="%.17g")
        control_profile_index.to_csv(staging / "TRAIN_CONTROL_PROFILE_INDEX.csv", index=False)
        pretruth_tasks = tasks.copy()
        pretruth_tasks["effect_asset_key"] = np.where(
            pretruth_tasks.task_id.isin(effects), pretruth_tasks.task_id, ""
        )
        pretruth_tasks["pretruth_x_access"] = np.where(
            pretruth_tasks.task_id.isin(effects), "SEEN_TARGET_EFFECT_AVAILABLE", "QUERY_ONLY"
        )
        pretruth_tasks.to_csv(staging / "PRETRUTH_TASKS.csv", index=False)
        access = pd.DataFrame(
            [
                {
                    "source_row_index": int(row.source_row_index),
                    "truth_access_phase": "PRETRUTH_CONTROL_X",
                    "logical_x_row_read_count": 1,
                    "asset_stage": "F2_PRETRUTH",
                    "purpose": "same_group_control_profile_and_panel_selection",
                }
                for row in control_rows.itertuples(index=False)
            ]
            + target_access
        ).sort_values("source_row_index", kind="stable")
        observed = access.truth_access_phase.value_counts().to_dict()
        expected = {
            phase: int(rows.truth_access_phase.eq(phase).sum())
            for phase in PRETRUTH_PHASES
        }
        if observed != expected:
            raise IntegrityError(f"F2 logical row access changed: {observed} != {expected}")
        forbidden_read = access.truth_access_phase.isin(FORBIDDEN_PHASES).sum()
        if int(forbidden_read) != 0:
            raise IntegrityError("forbidden calibration/evaluation row entered F2")
        access.to_csv(staging / "ROW_ACCESS_AUDIT.csv", index=False)

        primary_hashes = {
            path.name: sha256_file(path)
            for path in sorted(staging.iterdir())
            if path.is_file()
        }
        attestation = {
            "schema": "safeconf_e177_f2_asset_attestation_v1",
            "experiment": EXPERIMENT,
            "stage": "F2_PRETRUTH_ISOLATED_ASSET_BUILD",
            "status": "PASS",
            "generated_from_metadata_commit": FROZEN_METADATA_COMMIT,
            "current_git_head": frozen["head"],
            "code_freeze_branch": frozen["branch"],
            "code_freeze_remote_heads": frozen["remote_heads"],
            "builder_sha256": frozen["builder_sha256"],
            "frozen_input_sha256": frozen["frozen_hashes"],
            "source_path": str(source),
            "source_bytes": source.stat().st_size,
            "source_sha256": source_lock["source_sha256"],
            "source_x_encoding": "csc_matrix_backed_h5ad",
            "physical_hdf5_row_exactness_claimed": False,
            "logical_x_rows_read": int(len(access)),
            "logical_x_rows_read_by_phase": expected,
            "calibration_target_x_rows_read": 0,
            "evaluation_target_x_rows_read": 0,
            "n_panel_genes": int(len(panel)),
            "n_registered_targets_in_panel": int(panel.panel_role.eq("REGISTERED_TARGET").sum()),
            "n_control_profiles": int(len(controls)),
            "n_control_cells_used": int(len(control_rows)),
            "n_seen_train_validation_effects": int(len(effects)),
            "n_query_only_tasks_without_y": int(pretruth_tasks.effect_asset_key.eq("").sum()),
            "coexpression_uses_only_exact_control_cells": True,
            "public_processed_data_only": True,
            "operational_wetlab_protocol_in_scope": False,
            "primary_output_sha256": primary_hashes,
        }
        write_text(staging / "ACCESS_ATTESTATION.json", json.dumps(attestation, indent=2, sort_keys=True) + "\n")
        manifest_sha = write_manifest(staging)
        observed_files = {path.name for path in staging.iterdir() if path.is_file()}
        if observed_files != ALLOWLIST:
            raise IntegrityError(f"F2 allowlist mismatch: {sorted(observed_files)}")
        os.replace(staging, F2_DIR)
        return {
            "status": "PASS",
            "experiment": EXPERIMENT,
            "stage": "F2_PRETRUTH_ISOLATED_ASSET_BUILD",
            "output": str(F2_DIR),
            "manifest_sha256": manifest_sha,
            "logical_x_rows_read_by_phase": expected,
            "calibration_target_x_rows_read": 0,
            "evaluation_target_x_rows_read": 0,
            "n_seen_train_validation_effects": len(effects),
            "n_query_only_tasks_without_y": int(pretruth_tasks.effect_asset_key.eq("").sum()),
        }
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_existing() -> dict[str, Any]:
    if not F2_DIR.is_dir():
        raise IntegrityError(f"F2 asset directory does not exist: {F2_DIR}")
    observed_files = {path.name for path in F2_DIR.iterdir() if path.is_file()}
    if observed_files != ALLOWLIST:
        raise IntegrityError(f"F2 allowlist mismatch: {sorted(observed_files)}")
    manifest = {}
    for line in (F2_DIR / "MANIFEST.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        path = F2_DIR / name
        if not path.is_file() or sha256_file(path) != digest:
            raise IntegrityError(f"asset hash mismatch: {name}")
        manifest[name] = digest
    attestation = json.loads((F2_DIR / "ACCESS_ATTESTATION.json").read_text())
    if attestation.get("status") != "PASS":
        raise IntegrityError("F2 attestation is not PASS")
    return {
        "status": "PASS",
        "output": str(F2_DIR),
        "files": sorted(observed_files),
        "manifest_sha256": sha256_file(F2_DIR / "MANIFEST.sha256"),
        "logical_x_rows_read_by_phase": attestation["logical_x_rows_read_by_phase"],
        "calibration_target_x_rows_read": attestation["calibration_target_x_rows_read"],
        "evaluation_target_x_rows_read": attestation["evaluation_target_x_rows_read"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    result = validate_existing() if args.validate_only else build(args.batch_size)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
