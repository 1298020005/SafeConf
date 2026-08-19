#!/usr/bin/env python3
"""Shared sealed-data and geometry helpers for the E182 certificate."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/实验结果/E182_gse225807_registered_family_20260724"
F2_ROOT = Path("/home/yyf/data/safeconf_e182_gse225807/isolated/F2_pretruth")
SOURCE_LOCK = OUT / "SOURCE_LOCK.json"
TASKS = OUT / "manifests/E182_GUIDE_TASK_MANIFEST.csv"
PANEL = F2_ROOT / "GENE_PANEL.csv"
CONTROL = F2_ROOT / "CONTROL_PROFILES.npz"
PRETRUTH = OUT / "pretruth_release"
PREDICTIONS = PRETRUTH / "arrays/PRETRUTH_PREDICTIONS.npz"
SCORING = PRETRUTH / "tables/PRETRUTH_SCORING_INTERFACE.csv"
SNAPSHOT = PRETRUTH / "PRETRUTH_GATE_SNAPSHOT.json"
N_GENES = 512
SEEDS = (3407, 3408, 3409, 3410, 3411)
REGISTERED_NAMES = (
    *(f"scGPT_seed{seed}" for seed in SEEDS),
    *(f"GEARS_seed{seed}" for seed in SEEDS),
)


class IntegrityError(RuntimeError):
    """An E182 immutable input or truth-access boundary changed."""


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_bytes(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
    )


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_bytes(
        path,
        frame.to_csv(index=False, float_format="%.17g").encode(),
    )


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        np.savez_compressed(
            handle,
            **{
                key: np.asarray(value, dtype=np.float32)
                for key, value in sorted(arrays.items())
            },
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def git_text(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def verify_dual_remote(head: str) -> tuple[str, dict[str, str]]:
    branch = git_text("rev-parse", "--abbrev-ref", "HEAD")
    if branch == "HEAD":
        raise IntegrityError("E182 formal phase requires a named branch")
    result: dict[str, str] = {}
    for remote in ("origin", "github"):
        ref = f"refs/remotes/{remote}/{branch}"
        fetched = subprocess.run(
            [
                "git",
                "fetch",
                "--quiet",
                remote,
                f"refs/heads/{branch}:{ref}",
            ],
            cwd=ROOT,
            check=False,
        )
        if fetched.returncode:
            raise IntegrityError(f"cannot fetch E182 freeze from {remote}")
        remote_head = git_text("rev-parse", ref)
        if subprocess.run(
            ["git", "merge-base", "--is-ancestor", head, remote_head],
            cwd=ROOT,
            check=False,
        ).returncode:
            raise IntegrityError(f"E182 HEAD is absent from {remote}")
        result[remote] = remote_head
    return branch, result


def require_committed(paths: Iterable[Path]) -> dict[str, Any]:
    head = git_text("rev-parse", "HEAD")
    branch, remotes = verify_dual_remote(head)
    rows: list[dict[str, Any]] = []
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        local = path.read_bytes()
        try:
            committed = subprocess.check_output(
                ["git", "show", f"{head}:{relative}"], cwd=ROOT
            )
        except subprocess.CalledProcessError as exc:
            raise IntegrityError(f"uncommitted E182 input: {relative}") from exc
        if local != committed:
            raise IntegrityError(f"E182 input differs from HEAD: {relative}")
        rows.append(
            {
                "path": relative,
                "bytes": len(local),
                "sha256": hashlib.sha256(local).hexdigest(),
            }
        )
    return {
        "head": head,
        "branch": branch,
        "remote_heads": remotes,
        "input_hashes": rows,
    }


def parse_manifest(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        result[name.strip()] = digest
    return result


def validate_f2() -> list[dict[str, Any]]:
    manifest = F2_ROOT / "MANIFEST.sha256"
    if not manifest.is_file():
        raise IntegrityError("E182 F2 manifest is missing")
    rows: list[dict[str, Any]] = []
    for name, expected in parse_manifest(manifest).items():
        path = F2_ROOT / name
        if not path.is_file() or sha256_file(path) != expected:
            raise IntegrityError(f"E182 F2 hash mismatch: {name}")
        rows.append(
            {"path": str(path), "bytes": path.stat().st_size, "sha256": expected}
        )
    attestation = json.loads((F2_ROOT / "ACCESS_ATTESTATION.json").read_text())
    required = {
        "status": "PASS",
        "stage": "F2_PRETRUTH_ISOLATED_ASSET_BUILD",
        "calibration_target_x_rows_read": 0,
        "evaluation_target_x_rows_read": 0,
        "experiment": "E182_gse225807_registered_family",
    }
    if any(attestation.get(key) != value for key, value in required.items()):
        raise IntegrityError("E182 F2 access attestation changed")
    return rows


def load_pretruth() -> tuple[pd.DataFrame, dict[str, np.ndarray], list[dict[str, Any]]]:
    external_hashes = validate_f2()
    snapshot = json.loads(SNAPSHOT.read_text())
    if snapshot.get("status") != "PASS" or snapshot.get("adaptive_upper_fitted") is not False:
        raise IntegrityError("E182 pretruth snapshot is not eligible")
    scores = pd.read_csv(SCORING, keep_default_na=False)
    tasks = pd.read_csv(TASKS, keep_default_na=False)
    shared = [
        "task_id",
        "perturbation",
        "guide_id",
        "target_split",
        "target_cluster_id",
    ]
    left = scores[shared].sort_values("task_id").reset_index(drop=True).astype(str)
    right = tasks[shared].sort_values("task_id").reset_index(drop=True).astype(str)
    if not left.equals(right):
        raise IntegrityError("E182 pretruth interface differs from task manifest")
    arrays: dict[str, np.ndarray] = {}
    with np.load(PREDICTIONS, allow_pickle=False) as archive:
        expected = {*REGISTERED_NAMES, "scGPT_seed_mean", "GEARS_seed_mean", "registered_family_centroid"}
        if set(archive.files) != expected:
            raise IntegrityError("E182 prediction array family changed")
        for name in archive.files:
            value = np.asarray(archive[name], dtype=np.float32)
            if value.shape != (len(scores), N_GENES) or not np.isfinite(value).all():
                raise IntegrityError(f"invalid E182 prediction array: {name}")
            arrays[name] = value
    return scores, arrays, external_hashes


def normalize_log1p(values: Any) -> sp.csr_matrix:
    matrix = (
        values.tocsr().astype(np.float32)
        if sp.issparse(values)
        else sp.csr_matrix(np.asarray(values, dtype=np.float32))
    )
    totals = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float32)
    if not np.isfinite(totals).all() or (totals <= 0).any():
        raise IntegrityError("invalid E182 selected-cell library size")
    scale = np.divide(1.0e4, totals, out=np.zeros_like(totals), where=totals > 0)
    result = (sp.diags(scale, dtype=np.float32) @ matrix).tocsr()
    result.data = np.log1p(result.data)
    return result


def read_split_truth(
    split: str, batch_size: int = 1024
) -> tuple[dict[str, np.ndarray], pd.DataFrame, list[dict[str, Any]]]:
    if split not in {"conformal_calibration", "prospective_evaluation"}:
        raise IntegrityError(f"E182 hidden truth split is invalid: {split}")
    source_lock = json.loads(SOURCE_LOCK.read_text())
    source = Path(source_lock["source_path"])
    if not source.is_file() or sha256_file(source) != source_lock["source_sha256"]:
        raise IntegrityError("E182 source changed before hidden-truth access")
    panel = pd.read_csv(PANEL, keep_default_na=False)
    if len(panel) != N_GENES:
        raise IntegrityError("E182 panel size changed")
    columns = panel["source_column_index"].to_numpy(np.int64)
    with np.load(CONTROL, allow_pickle=False) as archive:
        control = np.asarray(archive["GLOBAL"], dtype=np.float32)
    if control.shape != (N_GENES,):
        raise IntegrityError("E182 control profile changed")

    all_tasks = pd.read_csv(TASKS, keep_default_na=False)
    selected_tasks = all_tasks[all_tasks["target_split"].eq(split)].copy()
    allowed = {
        (str(row.perturbation), str(row.guide_id)): str(row.task_id)
        for row in selected_tasks.itertuples(index=False)
    }
    if not allowed:
        raise IntegrityError(f"E182 has no tasks for {split}")

    adata = ad.read_h5ad(source, backed="r")
    access_rows: list[dict[str, Any]] = []
    sums: defaultdict[str, np.ndarray] = defaultdict(
        lambda: np.zeros(N_GENES, dtype=np.float64)
    )
    counts: defaultdict[str, int] = defaultdict(int)
    try:
        obs = adata.obs.copy()
        obs["perturbation"] = obs["perturbation"].astype(str)
        obs["guide_id"] = obs["guide_id"].astype(str)
        mask = np.fromiter(
            (
                (perturbation, guide) in allowed
                for perturbation, guide in zip(
                    obs["perturbation"], obs["guide_id"]
                )
            ),
            dtype=bool,
            count=len(obs),
        )
        rows = obs.loc[mask, ["perturbation", "guide_id"]].copy()
        rows["source_row_index"] = np.flatnonzero(mask)
        rows = rows.sort_values("source_row_index").reset_index(drop=True)
        for start in range(0, len(rows), batch_size):
            block = rows.iloc[start : start + batch_size]
            row_indices = block["source_row_index"].to_numpy(np.int64)
            values = adata.X[row_indices, :]
            normalized = normalize_log1p(values)[:, columns].toarray()
            if len(block) != normalized.shape[0]:
                raise IntegrityError(
                    "E182 hidden-truth metadata and expression batch lengths differ"
                )
            for meta, vector in zip(block.itertuples(index=False), normalized):
                task_id = allowed[(str(meta.perturbation), str(meta.guide_id))]
                sums[task_id] += vector
                counts[task_id] += 1
                access_rows.append(
                    {
                        "source_row_index": int(meta.source_row_index),
                        "task_id": task_id,
                        "perturbation": str(meta.perturbation),
                        "guide_id": str(meta.guide_id),
                        "target_split": split,
                        "logical_x_row_read_count": 1,
                    }
                )
    finally:
        adata.file.close()
    if set(sums) != set(selected_tasks["task_id"].astype(str)):
        raise IntegrityError(f"E182 {split} truth task set is incomplete")
    truth = {
        task_id: (sums[task_id] / counts[task_id] - control).astype(np.float32)
        for task_id in sorted(sums)
    }
    observed_counts = pd.Series(counts, name="n_cells").rename_axis("task_id")
    expected_counts = selected_tasks.set_index("task_id")["n_guide_cells"].astype(int)
    if not observed_counts.sort_index().equals(expected_counts.sort_index()):
        raise IntegrityError(f"E182 {split} row counts differ from frozen metadata")
    external = [
        {"path": str(source), "bytes": source.stat().st_size, "sha256": sha256_file(source)}
    ]
    return truth, pd.DataFrame(access_rows), external


def rms(values: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.sqrt(np.mean(np.square(values, dtype=np.float64), axis=axis))


def evaluate_tasks(
    scores: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    truth: dict[str, np.ndarray],
    split: str,
) -> pd.DataFrame:
    family = np.stack([arrays[name] for name in REGISTERED_NAMES], axis=0).astype(
        np.float64
    )
    centroid = family.mean(axis=0)
    stored_centroid = arrays["registered_family_centroid"].astype(np.float64)
    if np.max(np.abs(stored_centroid - centroid)) > 5e-7:
        raise IntegrityError("E182 stored family centroid differs from registered members")
    indices = scores.index[scores["target_split"].eq(split)].to_numpy(int)
    rows: list[dict[str, Any]] = []
    for index in indices:
        meta = scores.loc[index]
        task_id = str(meta["task_id"])
        y = truth[task_id].astype(np.float64)
        member_errors = rms(family[:, index, :] - y[None, :], axis=1)
        centroid_error = float(rms(centroid[index] - y))
        family_rms_error = float(np.sqrt(np.mean(np.square(member_errors))))
        diversity = float(meta["family_diversity_lower"])
        radius = float(meta["family_radius"])
        diameter = float(meta["family_diameter"])
        worst_lower = float(meta["worst_member_lower"])
        rows.append(
            {
                "task_id": task_id,
                "perturbation": str(meta["perturbation"]),
                "guide_id": str(meta["guide_id"]),
                "target_split": split,
                "n_guide_cells": int(meta["n_guide_cells"]),
                "centroid_rmse": centroid_error,
                "family_rms_error": family_rms_error,
                "family_diversity_lower": diversity,
                "family_lower_violation": diversity > family_rms_error + 1e-10,
                "worst_member_error": float(member_errors.max()),
                "worst_member_lower": worst_lower,
                "worst_member_lower_violation": worst_lower
                > float(member_errors.max()) + 1e-10,
                "family_radius": radius,
                "family_diameter": diameter,
                "hilbert_identity_residual": (
                    family_rms_error**2 - centroid_error**2 - diversity**2
                ),
                **{
                    f"{name}_rmse": float(member_errors[position])
                    for position, name in enumerate(REGISTERED_NAMES)
                },
            }
        )
    result = pd.DataFrame(rows)
    if set(result["task_id"]) != set(truth):
        raise IntegrityError(f"E182 {split} evaluated task set changed")
    return result
