#!/usr/bin/env python3
"""Build E180 F2 assets while keeping calibration/evaluation X values sealed."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_REL = Path("docs/实验结果/E180_xucao_fresh_guide_certificate_20260723")
EXPERIMENT = ROOT / EXPERIMENT_REL
FROZEN_METADATA_COMMIT = "b7cafe6e8bce8dc4d2e237f424605a89cf5d6fde"
SOURCE_LOCK_REL = EXPERIMENT_REL / "SOURCE_LOCK.json"
RUN_STATUS_REL = EXPERIMENT_REL / "RUN_STATUS.json"
MODEL_LOCK_REL = EXPERIMENT_REL / "MODEL_INPUT_LOCK.json"
STAT_LOCK_REL = EXPERIMENT_REL / "STATISTICAL_ANALYSIS_LOCK.json"
PLAN_REL = EXPERIMENT_REL / "PREREG_ANALYSIS_PLAN.md"
TARGETS_REL = EXPERIMENT_REL / "manifests/E180_SELECTED_TARGETS.csv"
TASKS_REL = EXPERIMENT_REL / "manifests/E180_GUIDE_TASK_MANIFEST.csv"
BUILDER_REL = Path("tools/scripts/build_e180_xucao_pretruth_assets.py")
FROZEN_INPUTS = (
    SOURCE_LOCK_REL,
    RUN_STATUS_REL,
    MODEL_LOCK_REL,
    STAT_LOCK_REL,
    PLAN_REL,
    TARGETS_REL,
    TASKS_REL,
)

DATA_ROOT = Path("/home/yyf/data/safeconf_e180_external")
ISOLATED_ROOT = DATA_ROOT / "isolated"
F2_DIR = ISOLATED_ROOT / "F2_pretruth"
N_PANEL_GENES = 512
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
    """A frozen input, immutable asset, or truth boundary changed."""


def sha256_file(path: Path, chunk_size: int = 64 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def git_text(*args: str) -> str:
    return git(*args).stdout.decode().strip()


def atomic_text(path: Path, value: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(value)
    os.replace(tmp, path)


def verify_remote_freeze(head: str) -> tuple[str, dict[str, str]]:
    branch = git_text("rev-parse", "--abbrev-ref", "HEAD")
    if branch == "HEAD":
        raise IntegrityError("E180 F2 requires a named branch")
    remote_heads: dict[str, str] = {}
    for remote in ("origin", "github"):
        ref = f"refs/remotes/{remote}/{branch}"
        result = git(
            "fetch", "--quiet", remote, f"refs/heads/{branch}:{ref}", check=False
        )
        if result.returncode:
            raise IntegrityError(f"cannot verify E180 code on {remote}")
        remote_head = git_text("rev-parse", ref)
        if git("merge-base", "--is-ancestor", head, remote_head, check=False).returncode:
            raise IntegrityError(f"E180 HEAD {head} is absent from {remote}/{branch}")
        remote_heads[remote] = remote_head
    return branch, remote_heads


def require_committed(relative: Path, head: str) -> str:
    path = ROOT / relative
    if not path.is_file():
        raise IntegrityError(f"missing required file: {relative}")
    committed = git("show", f"{head}:{relative.as_posix()}").stdout
    local = path.read_bytes()
    if committed != local:
        raise IntegrityError(f"working file differs from HEAD: {relative}")
    return hashlib.sha256(local).hexdigest()


def verify_frozen_state() -> dict[str, Any]:
    if git("cat-file", "-e", f"{FROZEN_METADATA_COMMIT}^{{commit}}", check=False).returncode:
        raise IntegrityError("E180 metadata-freeze commit is unavailable")
    head = git_text("rev-parse", "HEAD")
    if git("merge-base", "--is-ancestor", FROZEN_METADATA_COMMIT, head, check=False).returncode:
        raise IntegrityError("current HEAD does not descend from E180 metadata freeze")
    hashes: dict[str, str] = {}
    for relative in FROZEN_INPUTS:
        frozen = git("show", f"{FROZEN_METADATA_COMMIT}:{relative.as_posix()}").stdout
        local = (ROOT / relative).read_bytes()
        if frozen != local:
            raise IntegrityError(f"E180 frozen input changed: {relative}")
        hashes[relative.as_posix()] = hashlib.sha256(local).hexdigest()
    status = json.loads((ROOT / RUN_STATUS_REL).read_text())
    if status.get("status") != "F1_METADATA_FROZEN" or status.get("x_values_read_during_freeze") != 0:
        raise IntegrityError("E180 F1 freeze status is invalid")
    builder_sha = require_committed(BUILDER_REL, head)
    branch, remote_heads = verify_remote_freeze(head)
    return {
        "head": head,
        "branch": branch,
        "remote_heads": remote_heads,
        "frozen_hashes": hashes,
        "builder_sha256": builder_sha,
        "source_lock": json.loads((ROOT / SOURCE_LOCK_REL).read_text()),
        "model_lock": json.loads((ROOT / MODEL_LOCK_REL).read_text()),
    }


def normalize_log1p(values: Any) -> sp.csr_matrix:
    matrix = (
        values.tocsr().astype(np.float32)
        if sp.issparse(values)
        else sp.csr_matrix(np.asarray(values, dtype=np.float32))
    )
    totals = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float32)
    if not np.isfinite(totals).all() or (totals <= 0).any():
        raise IntegrityError("invalid library size in E180 selected rows")
    scale = np.divide(1.0e4, totals, out=np.zeros_like(totals), where=totals > 0)
    normalized = (sp.diags(scale, dtype=np.float32) @ matrix).tocsr()
    normalized.data = np.log1p(normalized.data)
    return normalized


def read_rows(
    adata: ad.AnnData, row_indices: Sequence[int], columns: np.ndarray | None = None
) -> sp.csr_matrix:
    rows = np.asarray(row_indices, dtype=np.int64)
    if len(rows) == 0 or len(np.unique(rows)) != len(rows):
        raise IntegrityError("E180 row read must be nonempty and unique")
    if (rows[1:] <= rows[:-1]).any():
        raise IntegrityError("E180 source rows must be strictly increasing")
    values = adata.X[rows, :]
    normalized = normalize_log1p(values)
    return normalized if columns is None else normalized[:, columns]


def save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    ordered = {key: np.asarray(value, np.float32) for key, value in sorted(arrays.items())}
    if not ordered:
        raise IntegrityError(f"empty E180 NPZ: {path}")
    np.savez_compressed(path, **ordered)
    with np.load(path, allow_pickle=False) as check:
        if set(check.files) != set(ordered):
            raise IntegrityError(f"E180 NPZ keys changed: {path}")
        for key, expected in ordered.items():
            if not np.array_equal(check[key], expected):
                raise IntegrityError(f"E180 NPZ round-trip changed: {path}:{key}")


def make_panel(
    var_names: list[str],
    vocab: set[str],
    targets: pd.DataFrame,
    control_mean: np.ndarray,
) -> pd.DataFrame:
    if len(var_names) != len(set(var_names)):
        raise IntegrityError("E180 expression var names are not unique")
    lookup = {gene: index for index, gene in enumerate(var_names)}
    rows: list[dict[str, Any]] = []
    used: set[str] = set()
    for target in targets.sort_values("selection_hash").itertuples(index=False):
        gene = str(target.perturbation)
        if gene not in lookup or gene not in vocab:
            raise IntegrityError(f"E180 registered target disappeared: {gene}")
        used.add(gene)
        rows.append(
            {
                "panel_index": len(rows),
                "source_column_index": lookup[gene],
                "gene_name": gene,
                "scgpt_token": gene,
                "panel_role": "REGISTERED_TARGET",
                "target_split": str(target.target_split),
                "control_mean_expression": float(control_mean[lookup[gene]]),
            }
        )
    for source_index in np.argsort(-control_mean, kind="stable"):
        gene = var_names[int(source_index)]
        if not gene or gene in used or gene not in vocab:
            continue
        used.add(gene)
        rows.append(
            {
                "panel_index": len(rows),
                "source_column_index": int(source_index),
                "gene_name": gene,
                "scgpt_token": gene,
                "panel_role": "CONTROL_HIGH_EXPRESSION",
                "target_split": "",
                "control_mean_expression": float(control_mean[int(source_index)]),
            }
        )
        if len(rows) == N_PANEL_GENES:
            break
    panel = pd.DataFrame(rows)
    if len(panel) != N_PANEL_GENES or panel["gene_name"].nunique() != N_PANEL_GENES:
        raise IntegrityError("E180 panel is incomplete or duplicated")
    if panel["panel_role"].eq("REGISTERED_TARGET").sum() != len(targets):
        raise IntegrityError("E180 target panel count changed")
    return panel


def build_coexpression(control_panel: np.ndarray, panel: pd.DataFrame) -> pd.DataFrame:
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = np.corrcoef(control_panel, rowvar=False)
    correlation = np.nan_to_num(np.abs(correlation), nan=0.0, posinf=0.0, neginf=0.0)
    genes = panel["scgpt_token"].astype(str).tolist()
    rows: list[dict[str, Any]] = []
    for target_index, target in enumerate(genes):
        candidates = np.argsort(correlation[:, target_index])[-11:][::-1].tolist()
        if target_index not in candidates:
            candidates.append(target_index)
        for source_index in candidates:
            weight = float(correlation[int(source_index), target_index])
            if weight >= 0.4 or int(source_index) == target_index:
                rows.append(
                    {
                        "source": genes[int(source_index)],
                        "target": target,
                        "importance": weight,
                        "source_panel_index": int(source_index),
                        "target_panel_index": target_index,
                        "rule": "absolute_control_correlation_top10_threshold_0.4_self_included",
                    }
                )
    edges = pd.DataFrame(rows)
    if set(genes) - set(edges["target"].astype(str)):
        raise IntegrityError("E180 coexpression graph misses panel genes")
    return edges


def aggregate_seen_effects(
    adata: ad.AnnData,
    obs: pd.DataFrame,
    tasks: pd.DataFrame,
    panel_columns: np.ndarray,
    control_profile: np.ndarray,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    allowed_tasks = tasks[
        tasks["target_split"].isin(["supervised_train", "model_validation"])
    ].copy()
    task_lookup = {
        (str(row.perturbation), str(row.guide_id)): str(row.task_id)
        for row in allowed_tasks.itertuples(index=False)
    }
    source_rows = obs[
        [
            (str(perturbation), str(guide)) in task_lookup
            for perturbation, guide in zip(obs["perturbation"], obs["guide_id"])
        ]
    ].copy()
    source_rows["source_row_index"] = np.flatnonzero(
        [
            (str(perturbation), str(guide)) in task_lookup
            for perturbation, guide in zip(obs["perturbation"], obs["guide_id"])
        ]
    )
    source_rows = source_rows.sort_values("source_row_index").reset_index(drop=True)
    sums: defaultdict[str, np.ndarray] = defaultdict(
        lambda: np.zeros(len(panel_columns), np.float64)
    )
    counts: defaultdict[str, int] = defaultdict(int)
    access: list[dict[str, Any]] = []
    for start in range(0, len(source_rows), batch_size):
        block = source_rows.iloc[start : start + batch_size]
        matrix = read_rows(
            adata, block["source_row_index"].astype(int).tolist(), panel_columns
        ).toarray()
        if len(block) != matrix.shape[0]:
            raise IntegrityError("E180 metadata and expression batch lengths differ")
        for meta, vector in zip(block.itertuples(index=False), matrix):
            task_id = task_lookup[(str(meta.perturbation), str(meta.guide_id))]
            sums[task_id] += vector
            counts[task_id] += 1
            split = str(
                allowed_tasks.set_index("task_id").loc[task_id, "target_split"]
            )
            access.append(
                {
                    "source_row_index": int(meta.source_row_index),
                    "truth_access_phase": (
                        "PRETRUTH_TRAIN_X"
                        if split == "supervised_train"
                        else "PRETRUTH_VALIDATION_X"
                    ),
                    "logical_x_row_read_count": 1,
                    "asset_stage": "F2_PRETRUTH",
                    "purpose": "registered_train_or_validation_guide_effect",
                }
            )
    effects = {
        task_id: (sums[task_id] / counts[task_id] - control_profile).astype(np.float32)
        for task_id in sorted(sums)
    }
    expected = set(allowed_tasks["task_id"].astype(str))
    if set(effects) != expected:
        raise IntegrityError(
            f"E180 seen effect tasks differ: missing={len(expected-set(effects))}"
        )
    return effects, access


def write_manifest(directory: Path) -> str:
    hashes = {
        path.name: sha256_file(path)
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name != "MANIFEST.sha256"
    }
    atomic_text(
        directory / "MANIFEST.sha256",
        "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())),
    )
    return sha256_file(directory / "MANIFEST.sha256")


def prepare_staging() -> Path:
    ISOLATED_ROOT.mkdir(parents=True, exist_ok=True)
    if F2_DIR.exists():
        raise IntegrityError(f"refusing to overwrite E180 F2: {F2_DIR}")
    stale = list(ISOLATED_ROOT.glob("F2_pretruth.staging.*"))
    if stale:
        raise IntegrityError(f"stale E180 staging directory: {stale}")
    staging = ISOLATED_ROOT / f"F2_pretruth.staging.{os.getpid()}"
    staging.mkdir(parents=True, mode=0o750)
    return staging


def build(batch_size: int) -> dict[str, Any]:
    frozen = verify_frozen_state()
    source_lock = frozen["source_lock"]
    source = Path(source_lock["source_path"])
    if not source.is_file() or sha256_file(source) != source_lock["source_sha256"]:
        raise IntegrityError("E180 source bytes changed")
    targets = pd.read_csv(ROOT / TARGETS_REL, keep_default_na=False)
    tasks = pd.read_csv(ROOT / TASKS_REL, keep_default_na=False)
    if targets["perturbation"].nunique() != len(targets):
        raise IntegrityError("E180 target manifest duplicated")
    if tasks.groupby("perturbation")["target_split"].nunique().max() != 1:
        raise IntegrityError("E180 target leaked across task splits")
    vocab_value = json.loads(
        Path(frozen["model_lock"]["scgpt_vocab"]["path"]).read_text()
    )
    vocab = set(vocab_value if isinstance(vocab_value, list) else vocab_value.keys())

    staging = prepare_staging()
    try:
        adata = ad.read_h5ad(source, backed="r")
        try:
            if [adata.n_obs, adata.n_vars] != source_lock["shape"]:
                raise IntegrityError("E180 source shape changed")
            obs = adata.obs.copy()
            obs["perturbation"] = obs["perturbation"].astype(str)
            obs["guide_id"] = obs["guide_id"].astype(str)
            control_indices = np.flatnonzero(obs["perturbation"].eq("control").to_numpy())
            control_norm = read_rows(adata, control_indices.tolist())
            control_mean_all = np.asarray(control_norm.mean(axis=0)).ravel()
            panel = make_panel(
                list(map(str, adata.var_names)), vocab, targets, control_mean_all
            )
            panel_columns = panel["source_column_index"].to_numpy(np.int64)
            control_panel = control_norm[:, panel_columns].toarray().astype(np.float32)
            control_profile = control_panel.mean(axis=0).astype(np.float32)
            coexpression = build_coexpression(control_panel, panel)
            effects, target_access = aggregate_seen_effects(
                adata,
                obs,
                tasks,
                panel_columns,
                control_profile,
                batch_size,
            )
        finally:
            adata.file.close()

        hidden = set(
            tasks.loc[
                tasks["target_split"].isin(
                    ["conformal_calibration", "prospective_evaluation"]
                ),
                "task_id",
            ].astype(str)
        )
        if set(effects) & hidden:
            raise IntegrityError("E180 hidden truth leaked into F2")

        panel.to_csv(staging / "GENE_PANEL.csv", index=False, float_format="%.17g")
        save_npz(staging / "CONTROL_PROFILES.npz", {"GLOBAL": control_profile})
        save_npz(staging / "SEEN_TARGET_EFFECTS.npz", effects)
        coexpression.to_csv(
            staging / "TRAIN_CONTROL_COEXPRESSION_EDGES.csv",
            index=False,
            float_format="%.17g",
        )
        pd.DataFrame(
            {
                "source_row_index": control_indices,
                "perturbation": "control",
                "used_for_control_profile": True,
                "used_for_panel_selection": True,
                "used_for_coexpression": True,
            }
        ).to_csv(staging / "TRAIN_CONTROL_PROFILE_INDEX.csv", index=False)
        pretruth_tasks = tasks.copy()
        pretruth_tasks["effect_asset_key"] = np.where(
            pretruth_tasks["task_id"].isin(effects), pretruth_tasks["task_id"], ""
        )
        pretruth_tasks["pretruth_x_access"] = np.where(
            pretruth_tasks["task_id"].isin(effects),
            "SEEN_TARGET_EFFECT_AVAILABLE",
            "QUERY_ONLY",
        )
        pretruth_tasks.to_csv(staging / "PRETRUTH_TASKS.csv", index=False)

        control_access = [
            {
                "source_row_index": int(index),
                "truth_access_phase": "PRETRUTH_CONTROL_X",
                "logical_x_row_read_count": 1,
                "asset_stage": "F2_PRETRUTH",
                "purpose": "pooled_control_panel_profile_and_coexpression",
            }
            for index in control_indices
        ]
        access = pd.DataFrame(control_access + target_access).sort_values(
            "source_row_index"
        )
        access.to_csv(staging / "ROW_ACCESS_AUDIT.csv", index=False)
        phase_counts = {
            key: int(value)
            for key, value in access["truth_access_phase"].value_counts().items()
        }
        if any("CALIBRATION" in key or "EVALUATION" in key for key in phase_counts):
            raise IntegrityError("E180 forbidden phase entered F2 access audit")

        primary_hashes = {
            path.name: sha256_file(path)
            for path in sorted(staging.iterdir())
            if path.is_file()
        }
        attestation = {
            "schema": "safeconf_e180_f2_asset_attestation_v1",
            "experiment": "E180_xucao_fresh_guide_certificate",
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
            "normalization": "per_cell_total_10000_then_log1p",
            "logical_x_rows_read": int(len(access)),
            "logical_x_rows_read_by_phase": phase_counts,
            "calibration_target_x_rows_read": 0,
            "evaluation_target_x_rows_read": 0,
            "n_panel_genes": len(panel),
            "n_registered_targets_in_panel": int(
                panel["panel_role"].eq("REGISTERED_TARGET").sum()
            ),
            "n_control_profiles": 1,
            "n_control_cells_used": len(control_indices),
            "n_seen_train_validation_effects": len(effects),
            "n_query_only_tasks_without_y": int(
                pretruth_tasks["effect_asset_key"].eq("").sum()
            ),
            "coexpression_uses_only_control_cells": True,
            "cell_cycle_phase_used_as_primary_context": False,
            "primary_output_sha256": primary_hashes,
        }
        atomic_text(
            staging / "ACCESS_ATTESTATION.json",
            json.dumps(attestation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        manifest_sha = write_manifest(staging)
        observed = {path.name for path in staging.iterdir() if path.is_file()}
        if observed != ALLOWLIST:
            raise IntegrityError(f"E180 F2 allowlist changed: {sorted(observed)}")
        os.replace(staging, F2_DIR)
        return {
            "status": "PASS",
            "stage": "F2_PRETRUTH_ISOLATED_ASSET_BUILD",
            "output": str(F2_DIR),
            "manifest_sha256": manifest_sha,
            "logical_x_rows_read_by_phase": phase_counts,
            "calibration_target_x_rows_read": 0,
            "evaluation_target_x_rows_read": 0,
            "n_seen_train_validation_effects": len(effects),
            "n_query_only_tasks_without_y": int(
                pretruth_tasks["effect_asset_key"].eq("").sum()
            ),
        }
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_existing() -> dict[str, Any]:
    if not F2_DIR.is_dir():
        raise IntegrityError(f"E180 F2 missing: {F2_DIR}")
    observed = {path.name for path in F2_DIR.iterdir() if path.is_file()}
    if observed != ALLOWLIST:
        raise IntegrityError(f"E180 F2 allowlist mismatch: {sorted(observed)}")
    for line in (F2_DIR / "MANIFEST.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        if sha256_file(F2_DIR / name) != digest:
            raise IntegrityError(f"E180 F2 hash mismatch: {name}")
    attestation = json.loads((F2_DIR / "ACCESS_ATTESTATION.json").read_text())
    if (
        attestation.get("status") != "PASS"
        or attestation.get("calibration_target_x_rows_read") != 0
        or attestation.get("evaluation_target_x_rows_read") != 0
    ):
        raise IntegrityError("E180 F2 attestation failed")
    return {
        "status": "PASS",
        "output": str(F2_DIR),
        "files": sorted(observed),
        "manifest_sha256": sha256_file(F2_DIR / "MANIFEST.sha256"),
        "logical_x_rows_read_by_phase": attestation["logical_x_rows_read_by_phase"],
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
