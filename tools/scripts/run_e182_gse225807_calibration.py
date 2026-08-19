#!/usr/bin/env python3
"""Open only E182 calibration targets and freeze the constant conformal upper."""

from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
COMMON_PATH = ROOT / "tools/scripts/e182_registered_family_common.py"
OUT = ROOT / "docs/实验结果/E182_gse225807_registered_family_20260724"
RELEASE = OUT / "calibration_release"
TARGET_COVERAGE = 0.90
COMMITTED_INPUTS = (
    RUNNER,
    COMMON_PATH,
    ROOT / "tools/scripts/run_e180_xucao_pretruth.py",
    ROOT / "tools/scripts/build_e180_xucao_pretruth_assets.py",
    ROOT / "tools/scripts/run_e182_gse225807_pretruth.py",
    ROOT / "tools/scripts/build_e182_gse225807_pretruth_assets.py",
    OUT / "SOURCE_LOCK.json",
    OUT / "MODEL_INPUT_LOCK.json",
    OUT / "STATISTICAL_ANALYSIS_LOCK.json",
    OUT / "PREREG_ANALYSIS_PLAN.md",
    OUT / "manifests/E182_SELECTED_TARGETS.csv",
    OUT / "manifests/E182_GUIDE_TASK_MANIFEST.csv",
    OUT / "pretruth_release/PRETRUTH_GATE_SNAPSHOT.json",
    OUT / "pretruth_release/tables/PRETRUTH_SCORING_INTERFACE.csv",
    OUT / "pretruth_release/tables/INPUT_HASHES.csv",
    OUT / "pretruth_release/arrays/PRETRUTH_PREDICTIONS.npz",
)


def import_common() -> Any:
    spec = importlib.util.spec_from_file_location("e182_calibration_common", COMMON_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {COMMON_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    common = import_common()
    if RELEASE.exists():
        raise common.IntegrityError("E182 calibration release is append-only")
    audit = common.require_committed(COMMITTED_INPUTS)
    scores, arrays, external_hashes = common.load_pretruth()
    truth, access, source_hashes = common.read_split_truth("conformal_calibration")
    errors = common.evaluate_tasks(
        scores, arrays, truth, "conformal_calibration"
    )
    if errors["family_lower_violation"].any():
        raise common.IntegrityError("E182 calibration family lower bound failed")
    if errors["worst_member_lower_violation"].any():
        raise common.IntegrityError("E182 calibration worst-member lower bound failed")
    if errors["hilbert_identity_residual"].abs().max() > 1e-10:
        raise common.IntegrityError("E182 calibration Hilbert identity failed")

    clusters = (
        errors.groupby("perturbation", observed=True)
        .agg(
            n_guides=("guide_id", "nunique"),
            max_centroid_rmse=("centroid_rmse", "max"),
            mean_centroid_rmse=("centroid_rmse", "mean"),
            max_family_rms_error=("family_rms_error", "max"),
            max_worst_member_error=("worst_member_error", "max"),
        )
        .reset_index()
        .sort_values("max_centroid_rmse")
        .reset_index(drop=True)
    )
    if clusters["n_guides"].ne(2).any():
        raise common.IntegrityError("E182 calibration target lost a guide")
    n_targets = len(clusters)
    rank_one_based = math.ceil((n_targets + 1) * TARGET_COVERAGE)
    if rank_one_based > n_targets:
        raise common.IntegrityError("E182 calibration size cannot support finite rank")
    correction = float(
        clusters["max_centroid_rmse"].sort_values().iloc[rank_one_based - 1]
    )
    clusters["finite_sample_rank_one_based"] = range(1, n_targets + 1)
    clusters["selected_as_conformal_quantile"] = (
        clusters["finite_sample_rank_one_based"].eq(rank_one_based)
    )

    staging = OUT / f".calibration_release.staging.{os.getpid()}"
    try:
        for subdirectory in ("tables", "arrays", "reports"):
            (staging / subdirectory).mkdir(parents=True, exist_ok=False)
        common.atomic_csv(staging / "tables/CALIBRATION_TASK_ERRORS.csv", errors)
        common.atomic_csv(staging / "tables/CALIBRATION_TARGET_CLUSTERS.csv", clusters)
        common.atomic_csv(staging / "tables/CALIBRATION_X_ACCESS_AUDIT.csv", access)
        common.atomic_csv(
            staging / "tables/INPUT_HASHES.csv",
            pd.DataFrame(
                audit["input_hashes"] + external_hashes + source_hashes
            ),
        )
        common.atomic_npz(staging / "arrays/CALIBRATION_TRUTH.npz", truth)
        lock = {
            "schema": "safeconf_e182_calibration_lock_v1",
            "status": "PASS",
            "experiment": "E182_gse225807_registered_family",
            "git_head": audit["head"],
            "git_branch": audit["branch"],
            "remote_heads": audit["remote_heads"],
            "target_coverage": TARGET_COVERAGE,
            "n_calibration_targets": n_targets,
            "n_calibration_tasks": len(errors),
            "finite_sample_rank_one_based": rank_one_based,
            "constant_centroid_upper": correction,
            "cluster_nonconformity": (
                "maximum registered-family centroid RMSE over both guides "
                "of one calibration target"
            ),
            "learned_or_adaptive_upper_fitted": False,
            "evaluation_target_x_rows_read": 0,
            "family_lower_violations": int(errors["family_lower_violation"].sum()),
            "worst_member_lower_violations": int(
                errors["worst_member_lower_violation"].sum()
            ),
            "max_hilbert_identity_absolute_residual": float(
                errors["hilbert_identity_residual"].abs().max()
            ),
            "logical_calibration_x_rows_read": len(access),
        }
        common.atomic_json(staging / "CALIBRATION_LOCK.json", lock)
        report = f"""# E182 calibration 报告

只打开了 {n_targets} 个校准靶基因、{len(errors)} 条 guide 任务。每个靶基因先取两条 guide 的家族质心误差最大值；90% 有限样本 split conformal 使用第 {rank_one_based} 个顺序统计量，冻结常数质心上界为 **{correction:.6f} RMSE**。

家族 RMS 下界违反 0 条，最坏成员下界违反 0 条，Hilbert 恒等式最大绝对残差为 `{lock['max_hilbert_identity_absolute_residual']:.3e}`。本阶段没有读取 20 个最终评价靶基因的表达值，也没有拟合学习型上界。
"""
        common.atomic_bytes(
            staging / "reports/E182_CALIBRATION_REPORT.md", report.encode()
        )
        os.replace(staging, RELEASE)
        print(json.dumps(lock, ensure_ascii=False, indent=2, sort_keys=True))
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
