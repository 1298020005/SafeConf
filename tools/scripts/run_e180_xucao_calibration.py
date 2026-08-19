#!/usr/bin/env python3
"""Open E180 calibration guides only and freeze target-cluster conformal corrections."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import importlib.util
import json
import math
import os
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
OUT = ROOT / "docs/实验结果/E180_xucao_fresh_guide_certificate_20260723"
RELEASE = OUT / "calibration_release"
STAGING = OUT / f".calibration_release.staging.{os.getpid()}"
PRETRUTH = OUT / "pretruth_release"
SOURCE_LOCK = OUT / "SOURCE_LOCK.json"
MODEL_LOCK = OUT / "MODEL_INPUT_LOCK.json"
STAT_LOCK = OUT / "STATISTICAL_ANALYSIS_LOCK.json"
TASKS = OUT / "manifests/E180_GUIDE_TASK_MANIFEST.csv"
BUILDER = ROOT / "tools/scripts/build_e180_xucao_pretruth_assets.py"
PRETRUTH_RUNNER = ROOT / "tools/scripts/run_e180_xucao_pretruth.py"
F2_ROOT = Path("/home/yyf/data/safeconf_e180_external/isolated/F2_pretruth")
TARGET_COVERAGE = 0.90
N_GENES = 512
METHOD_BASES = {
    "constant": "constant_base",
    "predicted_magnitude": "magnitude_base",
    "magnitude_plus_pair_lower": "magnitude_plus_lower_base",
    "extra_trees_vector": "extra_trees_vector_base",
}


class IntegrityError(RuntimeError):
    """E180 calibration integrity failure."""


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_bytes(path, frame.to_csv(index=False, float_format="%.17g").encode())


def atomic_json(path: Path, value: object) -> None:
    atomic_bytes(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
    )


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("xb") as handle:
        np.savez_compressed(
            handle,
            **{key: np.asarray(value, np.float32) for key, value in sorted(arrays.items())},
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def import_script(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise IntegrityError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def git_text(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def require_committed(path: Path, head: str) -> dict[str, Any]:
    relative = path.relative_to(ROOT).as_posix()
    local = path.read_bytes()
    try:
        committed = subprocess.check_output(["git", "show", f"{head}:{relative}"], cwd=ROOT)
    except subprocess.CalledProcessError as exc:
        raise IntegrityError(f"uncommitted E180 calibration input: {relative}") from exc
    if local != committed:
        raise IntegrityError(f"E180 calibration input differs from HEAD: {relative}")
    return {"path": relative, "bytes": len(local), "sha256": hashlib.sha256(local).hexdigest()}


def verify_remotes(head: str) -> tuple[str, dict[str, str]]:
    branch = git_text("rev-parse", "--abbrev-ref", "HEAD")
    result: dict[str, str] = {}
    for remote in ("origin", "github"):
        fetched = subprocess.run(
            [
                "git",
                "fetch",
                "--quiet",
                remote,
                f"refs/heads/{branch}:refs/remotes/{remote}/{branch}",
            ],
            cwd=ROOT,
            check=False,
        )
        if fetched.returncode:
            raise IntegrityError(f"cannot fetch E180 calibration freeze from {remote}")
        remote_head = git_text("rev-parse", f"refs/remotes/{remote}/{branch}")
        if subprocess.run(
            ["git", "merge-base", "--is-ancestor", head, remote_head],
            cwd=ROOT,
            check=False,
        ).returncode:
            raise IntegrityError(f"E180 calibration code absent from {remote}")
        result[remote] = remote_head
    return branch, result


def formal_audit() -> tuple[str, str, dict[str, str], list[dict[str, Any]]]:
    head = git_text("rev-parse", "HEAD")
    branch, remotes = verify_remotes(head)
    paths = [
        RUNNER,
        BUILDER,
        PRETRUTH_RUNNER,
        SOURCE_LOCK,
        MODEL_LOCK,
        STAT_LOCK,
        TASKS,
        PRETRUTH / "PRETRUTH_GATE_SNAPSHOT.json",
        PRETRUTH / "tables/PRETRUTH_SCORING_INTERFACE.csv",
        PRETRUTH / "arrays/PRETRUTH_PREDICTIONS.npz",
    ]
    hashes = [require_committed(path, head) for path in paths]
    snapshot = json.loads((PRETRUTH / "PRETRUTH_GATE_SNAPSHOT.json").read_text())
    if (
        snapshot.get("status") != "PASS"
        or snapshot.get("calibration_target_x_rows_read") != 0
        or snapshot.get("evaluation_target_x_rows_read") != 0
    ):
        raise IntegrityError("E180 pretruth snapshot is not eligible for calibration")
    return head, branch, remotes, hashes


def calibration_rank(n_targets: int) -> int:
    rank = math.ceil((n_targets + 1) * TARGET_COVERAGE)
    if rank > n_targets:
        raise IntegrityError("E180 calibration target count cannot support finite 90% rank")
    return rank


def aggregate_calibration_truth(
    source: Path,
    tasks: pd.DataFrame,
    panel: pd.DataFrame,
    control: np.ndarray,
    builder: Any,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    calibration = tasks[tasks["target_split"].eq("conformal_calibration")].copy()
    allowed = {
        (str(row.perturbation), str(row.guide_id)): str(row.task_id)
        for row in calibration.itertuples(index=False)
    }
    adata = ad.read_h5ad(source, backed="r")
    try:
        obs = adata.obs.copy()
        obs["perturbation"] = obs["perturbation"].astype(str)
        obs["guide_id"] = obs["guide_id"].astype(str)
        mask = np.asarray(
            [
                (perturbation, guide) in allowed
                for perturbation, guide in zip(
                    obs["perturbation"], obs["guide_id"], strict=True
                )
            ],
            dtype=bool,
        )
        rows = obs.loc[mask, ["perturbation", "guide_id"]].copy()
        rows["source_row_index"] = np.flatnonzero(mask)
        rows = rows.sort_values("source_row_index").reset_index(drop=True)
        panel_columns = panel["source_column_index"].to_numpy(np.int64)
        sums: defaultdict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(N_GENES, np.float64)
        )
        counts: defaultdict[str, int] = defaultdict(int)
        access: list[dict[str, Any]] = []
        for start in range(0, len(rows), batch_size):
            block = rows.iloc[start : start + batch_size]
            matrix = builder.read_rows(
                adata,
                block["source_row_index"].astype(int).tolist(),
                panel_columns,
            ).toarray()
            for meta, vector in zip(block.itertuples(index=False), matrix, strict=True):
                task_id = allowed[(str(meta.perturbation), str(meta.guide_id))]
                sums[task_id] += vector
                counts[task_id] += 1
                access.append(
                    {
                        "source_row_index": int(meta.source_row_index),
                        "task_id": task_id,
                        "truth_access_phase": "F3_CALIBRATION_ONLY",
                        "evaluation_target_x_read": False,
                    }
                )
    finally:
        adata.file.close()
    truth = {
        task_id: (sums[task_id] / counts[task_id] - control).astype(np.float32)
        for task_id in sorted(sums)
    }
    if set(truth) != set(calibration["task_id"].astype(str)):
        raise IntegrityError("E180 calibration truth task set incomplete")
    return truth, pd.DataFrame(access)


def main() -> None:
    if RELEASE.exists() or STAGING.exists():
        raise IntegrityError("E180 calibration release is append-only")
    audit = formal_audit()
    source_lock = json.loads(SOURCE_LOCK.read_text())
    source = Path(source_lock["source_path"])
    if sha256_file(source) != source_lock["source_sha256"]:
        raise IntegrityError("E180 source changed before calibration")
    builder = import_script("e180_builder_for_calibration", BUILDER)
    panel = pd.read_csv(F2_ROOT / "GENE_PANEL.csv")
    with np.load(F2_ROOT / "CONTROL_PROFILES.npz", allow_pickle=False) as archive:
        control = np.asarray(archive["GLOBAL"], np.float32)
    tasks = pd.read_csv(TASKS, keep_default_na=False)
    scores = pd.read_csv(PRETRUTH / "tables/PRETRUTH_SCORING_INTERFACE.csv")
    calibration_scores = scores[
        scores["target_split"].eq("conformal_calibration")
    ].copy()
    truth, access = aggregate_calibration_truth(
        source, tasks, panel, control, builder, batch_size=1024
    )
    evaluation_ids = set(
        tasks.loc[
            tasks["target_split"].eq("prospective_evaluation"), "task_id"
        ].astype(str)
    )
    if set(access["task_id"].astype(str)) & evaluation_ids:
        raise IntegrityError("E180 evaluation task entered calibration access")

    with np.load(PRETRUTH / "arrays/PRETRUTH_PREDICTIONS.npz", allow_pickle=False) as archive:
        scgpt = np.asarray(archive["scGPT_seed_mean"], np.float64)
        gears = np.asarray(archive["GEARS_seed_mean"], np.float64)
    row_index = {task: index for index, task in enumerate(scores["task_id"].astype(str))}
    error_rows: list[dict[str, Any]] = []
    for row in calibration_scores.itertuples(index=False):
        task_id = str(row.task_id)
        index = row_index[task_id]
        target = np.asarray(truth[task_id], np.float64)
        sc_error = float(np.sqrt(np.mean((scgpt[index] - target) ** 2)))
        ge_error = float(np.sqrt(np.mean((gears[index] - target) ** 2)))
        disagreement = float(np.sqrt(np.mean((scgpt[index] - gears[index]) ** 2)))
        item = {
            "task_id": task_id,
            "perturbation": str(row.perturbation),
            "guide_id": str(row.guide_id),
            "n_guide_cells": int(row.n_guide_cells),
            "scgpt_rmse": sc_error,
            "gears_rmse": ge_error,
            "pair_mean_rmse": (sc_error + ge_error) / 2.0,
            "pair_max_rmse": max(sc_error, ge_error),
            "model_disagreement_rmse": disagreement,
            "pair_lower_bound": disagreement / 2.0,
        }
        for method, column in METHOD_BASES.items():
            item[f"base__{method}"] = float(getattr(row, column))
        error_rows.append(item)
    errors = pd.DataFrame(error_rows)
    if (
        (errors["pair_lower_bound"] > errors["pair_mean_rmse"] + 1e-9).any()
        or (errors["pair_lower_bound"] > errors["pair_max_rmse"] + 1e-9).any()
    ):
        raise IntegrityError("E180 deterministic lower bound failed in calibration")

    n_targets = errors["perturbation"].nunique()
    rank = calibration_rank(n_targets)
    residual_rows: list[dict[str, Any]] = []
    models: dict[str, Any] = {}
    for method in METHOD_BASES:
        work = errors[["perturbation", "pair_mean_rmse"]].copy()
        work["residual"] = (
            errors["pair_mean_rmse"].to_numpy()
            - errors[f"base__{method}"].to_numpy()
        )
        cluster = (
            work.groupby("perturbation")["residual"].max().sort_values().rename(
                "cluster_residual_max"
            )
        )
        correction = float(cluster.iloc[rank - 1])
        models[method] = {
            "base_column": METHOD_BASES[method],
            "target_cluster_count": n_targets,
            "finite_sample_rank_one_based": rank,
            "target_coverage": TARGET_COVERAGE,
            "conformal_correction": correction,
        }
        for perturbation, value in cluster.items():
            residual_rows.append(
                {
                    "perturbation": perturbation,
                    "method": method,
                    "cluster_residual_max": value,
                    "selected_quantile_boundary": bool(
                        math.isclose(value, correction, rel_tol=0, abs_tol=1e-12)
                    ),
                }
            )
    model = {
        "schema": "safeconf_e180_target_cluster_conformal_model_v1",
        "status": "FROZEN",
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "primary_outcome": "pair_mean_rmse",
        "cluster_unit": "perturbation_gene_all_eligible_guides",
        "primary_method": "extra_trees_vector",
        "selection_changed_after_e180_truth": False,
        "methods": models,
        "calibration_pair_lower_mean_violations": int(
            (errors["pair_lower_bound"] > errors["pair_mean_rmse"] + 1e-9).sum()
        ),
        "calibration_pair_lower_max_violations": int(
            (errors["pair_lower_bound"] > errors["pair_max_rmse"] + 1e-9).sum()
        ),
        "evaluation_truth_opened": False,
    }

    try:
        for sub in ("tables", "arrays", "reports"):
            (STAGING / sub).mkdir(parents=True, exist_ok=False)
        atomic_csv(STAGING / "tables/CALIBRATION_TASK_ERRORS.csv", errors)
        atomic_csv(
            STAGING / "tables/CALIBRATION_RESIDUALS_BY_TARGET.csv",
            pd.DataFrame(residual_rows),
        )
        atomic_csv(STAGING / "tables/CALIBRATION_ROW_ACCESS_AUDIT.csv", access)
        atomic_npz(STAGING / "arrays/CALIBRATION_TRUE_EFFECTS.npz", truth)
        head, branch, remotes, hashes = audit
        input_hashes = hashes + [
            {
                "path": str(F2_ROOT / name),
                "bytes": (F2_ROOT / name).stat().st_size,
                "sha256": sha256_file(F2_ROOT / name),
            }
            for name in (
                "GENE_PANEL.csv",
                "CONTROL_PROFILES.npz",
                "MANIFEST.sha256",
            )
        ]
        atomic_csv(STAGING / "tables/INPUT_HASHES.csv", pd.DataFrame(input_hashes))
        atomic_json(STAGING / "CALIBRATION_MODEL.json", model)
        attestation = {
            "schema": "safeconf_e180_calibration_access_v1",
            "status": "PASS",
            "git_head": head,
            "git_branch": branch,
            "remote_heads": remotes,
            "n_calibration_targets": n_targets,
            "n_calibration_tasks": len(errors),
            "calibration_x_rows_read": len(access),
            "evaluation_target_x_rows_read": 0,
            "evaluation_truth_opened": False,
        }
        atomic_json(STAGING / "ACCESS_ATTESTATION.json", attestation)
        report = (
            "# E180 calibration 报告\n\n"
            f"已打开 {n_targets} 个 calibration 基因、{len(errors)} 个 guide 任务；"
            f"evaluation 表达读取仍为 0。\n\n"
            f"90% 靶点同时覆盖使用第 {rank}/{n_targets} 个有序靶点最大残差。"
            f"主方法 ExtraTrees 的冻结修正量为 "
            f"`{models['extra_trees_vector']['conformal_correction']:.6f}` RMSE。"
            "四种方法全部原样保留，未依据 calibration 结果更换主方法。\n\n"
            "下一步必须先提交并双远端推送本 calibration 模型，再打开 evaluation 真值。\n"
        )
        atomic_bytes(STAGING / "reports/E180_CALIBRATION_REPORT.md", report.encode())
        os.replace(STAGING, RELEASE)
    except Exception:
        shutil.rmtree(STAGING, ignore_errors=True)
        raise
    print(json.dumps(model, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
