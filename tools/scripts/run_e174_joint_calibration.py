#!/usr/bin/env python3
"""Calibrate E174 upper-risk bounds without opening final evaluation truth."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any
import uuid

import numpy as np
import pandas as pd

from e174_conformal_common import (
    MODEL_SPECS,
    add_pair_columns,
    calibrate_cluster_upper,
    load_npz_vectors,
    predict_ridge,
    rmse_rows,
    target_key,
)


ROOT = Path(__file__).resolve().parents[2]
JOINT_HELPER_PATH = ROOT / "tools/scripts/run_e170_primary_cd4_joint_postgate.py"
EXPERIMENT = ROOT / "docs/实验结果/E174_rotated_donor_conformal_certificate_20260719"
DATA_ROOT = Path("/home/yyf/data/safeconf_external/primary_cd4_perturbseq_2025")
OUT = EXPERIMENT / "calibration_release"
STAGING = EXPERIMENT / ".calibration_release.staging"
METHOD_SNAPSHOT = EXPERIMENT / "method_development/METHOD_GATE_SNAPSHOT.json"
METHOD_STATUS = EXPERIMENT / "method_development/RUN_STATUS.json"
METHOD_MANIFEST = EXPERIMENT / "method_development/MANIFEST.sha256"
PANELS = ("R01", "R02", "R03", "R04")
SCRIPT = Path(__file__).resolve()
COMMON = ROOT / "tools/scripts/e174_conformal_common.py"
PRETRUTH_WRAPPER = ROOT / "tools/scripts/run_e174_rotated_donor_panel_pretruth.py"
PRETRUTH_HELPER = ROOT / "tools/scripts/run_e168_primary_cd4_pretruth.py"
ASSET_WRAPPER = ROOT / "tools/scripts/build_e174_rotated_donor_panel_assets.py"
ASSET_HELPER = ROOT / "tools/scripts/build_e168_primary_cd4_isolated_assets.py"
FREEZE_SCRIPT = ROOT / "tools/scripts/freeze_e174_rotated_donor_conformal_certificate.py"
METHOD_SCRIPT = ROOT / "tools/scripts/run_e174_method_development.py"
FINAL_EVALUATOR = ROOT / "tools/scripts/run_e174_final_evaluation.py"
EXECUTION_PLAN = EXPERIMENT / "EXECUTION_PLAN.md"
STAT_LOCK = EXPERIMENT / "STATISTICAL_ANALYSIS_LOCK.json"
MODEL_LOCK = EXPERIMENT / "MODEL_INPUT_LOCK.json"
ALL_TARGETS = EXPERIMENT / "manifests/E174_ALL_SELECTED_TARGETS.csv"
ALL_TASKS = EXPERIMENT / "manifests/E174_ALL_TASKS.csv"
F2_ALLOWLIST = {
    "GENE_PANEL.csv",
    "CONTROL_PROFILES.npz",
    "SEEN_TARGET_EFFECTS.npz",
    "PRETRUTH_TASKS.csv",
    "PRETRUTH_GUIDE_EFFECT_INDEX.csv",
    "TRAIN_NTC_COEXPRESSION_EDGES.csv",
    "TRAIN_NTC_COEXPRESSION_PROFILE_INDEX.csv",
    "ROW_ACCESS_AUDIT.csv",
    "ACCESS_ATTESTATION.json",
    "MANIFEST.sha256",
}
F3_ALLOWLIST = {
    "CALIBRATION_TARGET_EFFECTS.npz",
    "CALIBRATION_GUIDE_EFFECTS.npz",
    "CALIBRATION_GUIDE_EFFECT_INDEX.csv",
    "CALIBRATION_TASKS.csv",
    "ROW_ACCESS_AUDIT.csv",
    "ACCESS_ATTESTATION.json",
    "MANIFEST.sha256",
}


class IntegrityFailure(RuntimeError):
    pass


def import_joint_helper() -> Any:
    spec = importlib.util.spec_from_file_location("safeconf_e174_calibration_helper", JOINT_HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import joint helper: {JOINT_HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.EXPERIMENT_CODE = "E174"
    module.EXPERIMENT_ID = "E174_rotated_donor_conformal_certificate"
    module.ISOLATED_NAMESPACE = "E174"
    module.TASK_PREFIX = "E174"
    module.PRETRUTH_SCHEMA_PREFIX = "safeconf_e174"
    module.PRETRUTH_REPORT_PREFIX = "E174"
    module.EXPECTED_G4_RISK_ESTIMATOR = "leave_one_seed_out_family_mean"
    module.EXPERIMENT = EXPERIMENT
    module.DATA_ROOT = DATA_ROOT
    module.PANELS = PANELS
    module.PRETRUTH_WRAPPER = PRETRUTH_WRAPPER
    module.PRETRUTH_HELPER = PRETRUTH_HELPER
    module.ASSET_WRAPPER = ASSET_WRAPPER
    module.ASSET_HELPER = ASSET_HELPER
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )


def git_text(*args: str) -> str:
    return git(*args).stdout.decode().strip()


def require_bytes_at_commit(path: Path, commit: str) -> dict[str, Any]:
    relative = path.relative_to(ROOT).as_posix()
    try:
        committed = git("show", f"{commit}:{relative}").stdout
    except subprocess.CalledProcessError as exc:
        raise IntegrityFailure(f"file absent from code freeze: {relative}") from exc
    if committed != path.read_bytes():
        raise IntegrityFailure(f"file changed after code freeze: {relative}")
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def verify_code_freeze(gate_commit: str, branch: str) -> tuple[dict[str, str], list[dict[str, Any]]]:
    if git("merge-base", "--is-ancestor", gate_commit, "HEAD", check=False).returncode:
        raise IntegrityFailure("current HEAD does not descend from pretruth gate commit")
    files = [
        SCRIPT,
        COMMON,
        JOINT_HELPER_PATH,
        PRETRUTH_WRAPPER,
        PRETRUTH_HELPER,
        ASSET_WRAPPER,
        ASSET_HELPER,
        FREEZE_SCRIPT,
        METHOD_SCRIPT,
        FINAL_EVALUATOR,
        EXECUTION_PLAN,
        STAT_LOCK,
        MODEL_LOCK,
        ALL_TARGETS,
        ALL_TASKS,
        METHOD_SNAPSHOT,
        METHOD_STATUS,
        METHOD_MANIFEST,
    ]
    hashes = [require_bytes_at_commit(path, gate_commit) for path in files]
    remote_heads: dict[str, str] = {}
    for remote in ("origin", "github"):
        fetched = f"refs/remotes/{remote}/{branch}"
        result = git("fetch", "--quiet", remote, f"refs/heads/{branch}:{fetched}", check=False)
        if result.returncode:
            raise IntegrityFailure(f"cannot verify pretruth gate on {remote}")
        remote_head = git_text("rev-parse", fetched)
        if git("merge-base", "--is-ancestor", gate_commit, remote_head, check=False).returncode:
            raise IntegrityFailure(f"pretruth gate absent from {remote}/{branch}")
        remote_heads[remote] = remote_head
    return remote_heads, hashes


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode())


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_bytes(path, frame.to_csv(index=False, float_format="%.17g").encode())


def strict_flag(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    values = series.astype(str).str.lower()
    if not values.isin({"true", "false"}).all():
        raise IntegrityFailure("non-boolean frozen task flag")
    return values.eq("true")


def load_calibration_panel(
    panel: str, gate_commit: str, helper: Any
) -> tuple[pd.DataFrame, dict[str, str], list[dict[str, Any]]]:
    snapshot, release = helper.verify_pretruth_release(panel, gate_commit)
    isolated = DATA_ROOT / "isolated/E174" / panel
    f2, f3 = isolated / "F2_pretruth", isolated / "F3A_calibration"
    _f2_files, f2_manifest = helper.parse_flat_manifest(f2, F2_ALLOWLIST)
    _f3_files, f3_manifest = helper.parse_flat_manifest(f3, F3_ALLOWLIST)
    if snapshot.get("f2_manifest_sha256") != f2_manifest:
        raise IntegrityFailure(f"{panel} pretruth snapshot does not bind F2")
    f2_att = json.loads((f2 / "ACCESS_ATTESTATION.json").read_text())
    f3_att = json.loads((f3 / "ACCESS_ATTESTATION.json").read_text())
    required_f3 = {
        "experiment": f"E174_rotated_donor_conformal_certificate::{panel}",
        "stage": f"E174_{panel}_F3A_CALIBRATION_TRUTH_BUILD",
        "status": "PASS",
        "f2_manifest_sha256": f2_manifest,
        "source_full_sha256": snapshot["source_full_sha256"],
        "gate_commit": gate_commit,
        "calibration_targeting_x_values_read": 240,
        "evaluation_targeting_x_values_read": 0,
        "other_truth_partition_x_values_read": 0,
        "n_calibration_target_effects": 120,
        "n_calibration_guide_effects": 240,
        "test_performance_metrics_computed": 0,
    }
    mismatches = {
        key: {"expected": value, "observed": f3_att.get(key)}
        for key, value in required_f3.items()
        if f3_att.get(key) != value
    }
    if mismatches:
        raise IntegrityFailure(f"{panel} F3A attestation failed: {mismatches}")
    if f2_att.get("source_full_sha256") != f3_att.get("source_full_sha256"):
        raise IntegrityFailure(f"{panel} F2/F3A source differs")
    if f2_att.get("builder_sha256") != sha256_file(ASSET_WRAPPER):
        raise IntegrityFailure(f"{panel} F2 builder changed")
    if f3_att.get("builder_sha256") != sha256_file(ASSET_WRAPPER):
        raise IntegrityFailure(f"{panel} F3A builder changed")

    scoring = pd.read_csv(release / "tables/PRETRUTH_SCORING_INTERFACE.csv", keep_default_na=False)
    if len(scoring) != 2160 or scoring.task_id.nunique() != 2160:
        raise IntegrityFailure(f"{panel} scoring interface changed")
    locked = pd.read_csv(
        EXPERIMENT / f"manifests/{panel}/E174_{panel}_TASK_MANIFEST.csv",
        keep_default_na=False,
    )
    calibration_tasks = locked.loc[strict_flag(locked.calibration_test_task)].copy()
    calibration_ids = calibration_tasks.task_id.astype(str).tolist()
    if len(calibration_ids) != 120:
        raise IntegrityFailure(f"{panel} calibration task count changed")
    score_index = scoring.set_index("task_id")
    if not set(calibration_ids).issubset(score_index.index.astype(str)):
        raise IntegrityFailure(f"{panel} calibration query missing from pretruth predictions")
    metadata = score_index.loc[calibration_ids].reset_index()
    if metadata.heldout_donor_partition.astype(str).ne("CALIBRATION_20PCT").any():
        raise IntegrityFailure(f"{panel} non-calibration task entered F3A")

    with np.load(release / "arrays/PRETRUTH_PREDICTIONS.npz", allow_pickle=False) as archive:
        row = {task: index for index, task in enumerate(scoring.task_id.astype(str))}
        prediction = {
            name: np.stack([np.asarray(archive[name][row[task]], dtype=float) for task in calibration_ids])
            for name in ("scGPT_seed_mean", "GEARS_seed_mean", "ensemble_seed_family_mean")
        }
    truth = load_npz_vectors(f3 / "CALIBRATION_TARGET_EFFECTS.npz", 120)
    if set(truth) != set(calibration_ids):
        raise IntegrityFailure(f"{panel} calibration truth/query keys differ")
    truth_matrix = np.stack([truth[task] for task in calibration_ids])
    sc = prediction["scGPT_seed_mean"]
    ge = prediction["GEARS_seed_mean"]
    ensemble = prediction["ensemble_seed_family_mean"]
    disagreement = rmse_rows(sc, ge)
    if not np.allclose(
        disagreement, metadata.model_disagreement_rmse.to_numpy(float), atol=2e-7, rtol=2e-6
    ):
        raise IntegrityFailure(f"{panel} disagreement differs from frozen scoring interface")
    sc_error, ge_error = rmse_rows(sc, truth_matrix), rmse_rows(ge, truth_matrix)
    metrics = metadata.copy()
    metrics["panel_id"] = panel
    metrics["ensemble_rmse"] = rmse_rows(ensemble, truth_matrix)
    metrics["scgpt_rmse"] = sc_error
    metrics["gears_rmse"] = ge_error
    metrics["pair_mean_rmse"] = (sc_error + ge_error) / 2.0
    metrics["pair_max_rmse"] = np.maximum(sc_error, ge_error)
    metrics["pair_lower_bound_rmse"] = disagreement / 2.0
    metrics["pair_mean_mse"] = (sc_error**2 + ge_error**2) / 2.0
    metrics["decomposition_rhs_mse"] = metrics.ensemble_rmse**2 + disagreement**2 / 4.0
    metrics["decomposition_abs_residual"] = np.abs(
        metrics.pair_mean_mse - metrics.decomposition_rhs_mse
    )
    metrics["nochange_rmse"] = np.sqrt(np.mean(truth_matrix**2, axis=1))
    hashes = [
        {"panel_id": panel, "path": str(path), "sha256": sha256_file(path)}
        for path in (
            f2 / "MANIFEST.sha256",
            f3 / "MANIFEST.sha256",
            release / "PRETRUTH_GATE_SNAPSHOT.json",
        )
    ]
    return metrics, {
        "source_full_sha256": snapshot["source_full_sha256"],
        "f2_manifest_sha256": f2_manifest,
        "f3_manifest_sha256": f3_manifest,
    }, hashes


def flatten_rules(rules: dict[str, dict[str, dict[str, Any]]]) -> pd.DataFrame:
    rows = []
    for outcome, by_spec in rules.items():
        for spec, rule in by_spec.items():
            rows.append({"outcome": outcome, "model_spec": spec, **rule})
    return pd.DataFrame(rows)


def write_manifest() -> str:
    files = sorted(
        path for path in STAGING.rglob("*") if path.is_file() and path.name not in {"MANIFEST.sha256", "RUN_STATUS.json"}
    )
    text = "".join(
        f"{sha256_file(path)}  {path.relative_to(STAGING).as_posix()}\n" for path in files
    )
    atomic_bytes(STAGING / "MANIFEST.sha256", text.encode())
    return sha256_file(STAGING / "MANIFEST.sha256")


def run(gate_commit: str, branch: str) -> dict[str, Any]:
    started = time.time()
    if OUT.exists() or STAGING.exists():
        raise IntegrityFailure("E174 calibration output is append-only")
    remote_heads, code_hashes = verify_code_freeze(gate_commit, branch)
    method = json.loads(METHOD_SNAPSHOT.read_text())
    method_status = json.loads(METHOD_STATUS.read_text())
    required_method = {
        "schema": "safeconf_e174_method_gate_snapshot_v1",
        "status": "PASS",
        "e174_expression_x_values_read": 0,
        "e174_calibration_truth_used": False,
        "e174_evaluation_truth_used": False,
    }
    if any(method.get(key) != value for key, value in required_method.items()):
        raise IntegrityFailure("method gate changed")
    if method.get("selected_model_spec") != {
        "ensemble_rmse": "magnitude",
        "pair_mean_rmse": "magnitude",
    }:
        raise IntegrityFailure("development fallback selection changed")
    if sha256_file(METHOD_MANIFEST) != method_status.get("manifest_sha256"):
        raise IntegrityFailure("method development manifest changed")

    helper = import_joint_helper()
    panel_metadata: dict[str, dict[str, str]] = {}
    metrics_list, input_hashes = [], []
    for panel in PANELS:
        metrics, metadata, hashes = load_calibration_panel(panel, gate_commit, helper)
        metrics_list.append(metrics)
        panel_metadata[panel] = metadata
        input_hashes.extend(hashes)
    metrics = add_pair_columns(pd.concat(metrics_list, ignore_index=True))
    if len(metrics) != 480 or target_key(metrics).nunique() != 160:
        raise IntegrityFailure("joint calibration population must be 160 targets / 480 tasks")
    if metrics.heldout_donor_partition.astype(str).ne("CALIBRATION_20PCT").any():
        raise IntegrityFailure("evaluation target entered calibration metrics")
    if int((metrics.pair_lower_bound_rmse > metrics.pair_mean_rmse + 1e-10).sum()) != 0:
        raise IntegrityFailure("pair-mean certificate violation in calibration")
    if int((metrics.pair_lower_bound_rmse > metrics.pair_max_rmse + 1e-10).sum()) != 0:
        raise IntegrityFailure("pair-max certificate violation in calibration")
    max_identity_residual = float(metrics.decomposition_abs_residual.max())
    if max_identity_residual > 1e-8:
        raise IntegrityFailure("squared-error identity failed in calibration")

    rules: dict[str, dict[str, dict[str, Any]]] = {}
    for outcome in ("ensemble_rmse", "pair_mean_rmse"):
        rules[outcome] = {}
        for spec in MODEL_SPECS:
            model = method["models"][outcome][spec]
            base = predict_ridge(metrics, model)
            rule = calibrate_cluster_upper(metrics, base, outcome, 0.90)
            rules[outcome][spec] = rule
            metrics[f"base_{outcome}__{spec}"] = base

    selected_rules = {
        outcome: rules[outcome][method["selected_model_spec"][outcome]]
        for outcome in ("ensemble_rmse", "pair_mean_rmse")
    }
    source_hashes = {value["source_full_sha256"] for value in panel_metadata.values()}
    if len(source_hashes) != 1:
        raise IntegrityFailure("panels were not built from one source object")

    for sub in ("tables", "reports"):
        (STAGING / sub).mkdir(parents=True, exist_ok=False)
    atomic_csv(STAGING / "tables/CALIBRATION_TASK_METRICS.csv", metrics)
    atomic_csv(STAGING / "tables/CALIBRATION_RULES.csv", flatten_rules(rules))
    atomic_csv(STAGING / "tables/INPUT_HASHES.csv", pd.DataFrame(code_hashes + input_hashes))
    payload_hashes = {
        path.relative_to(STAGING).as_posix(): sha256_file(path)
        for path in sorted(STAGING.rglob("*"))
        if path.is_file()
    }
    snapshot = {
        "schema": "safeconf_e174_joint_calibration_snapshot_v1",
        "experiment": "E174_rotated_donor_conformal_certificate",
        "stage": "F3A_CONFORMAL_CALIBRATION_GATE",
        "status": "PASS",
        "pretruth_gate_commit": gate_commit,
        "pretruth_gate_remote_heads": remote_heads,
        "source_full_sha256": next(iter(source_hashes)),
        "f2_manifest_sha256": {
            panel: panel_metadata[panel]["f2_manifest_sha256"] for panel in PANELS
        },
        "f3_calibration_manifest_sha256": {
            panel: panel_metadata[panel]["f3_manifest_sha256"] for panel in PANELS
        },
        "method_snapshot_sha256": sha256_file(METHOD_SNAPSHOT),
        "selected_model_spec": method["selected_model_spec"],
        "models": method["models"],
        "calibration_rules": rules,
        "selected_calibration_rules": selected_rules,
        "n_calibration_targets": 160,
        "n_calibration_tasks": 480,
        "calibration_targeting_x_values_read": 960,
        "evaluation_targeting_x_values_read": 0,
        "method_frozen_before_evaluation_truth": True,
        "pair_mean_bound_violations": 0,
        "pair_max_bound_violations": 0,
        "squared_error_decomposition_max_abs_residual": max_identity_residual,
        "calibration_payload_sha256": payload_hashes,
        "deployment_authorized": False,
    }
    atomic_json(STAGING / "CALIBRATION_GATE_SNAPSHOT.json", snapshot)
    report = f"""# E174 calibration gate

四个 pretruth gate 均已提交并通过后，只开放预注册的 160 个校准靶点、480 个任务；最终 640 个评价靶点 targeting X 读取数仍为 **0**。

旧数据方法门已冻结选择 magnitude 作为 ensemble RMSE 与 pair-mean RMSE 的基础估计器。校准阶段只计算 target-cluster residual 的有限样本第 145 顺序统计量，不重新拟合特征、不切换模型。pair mean/max 下界违例均为 0，平方误差分解最大残差为 {max_identity_residual:.3g}。

本 snapshot 与四个 F2、四个 F3A manifest、方法 snapshot 和源文件 SHA-256 绑定。只有把它提交到 GitHub 与 Gitee 后，F4 builder 才能读取剩余评价真值。
"""
    atomic_bytes(STAGING / "reports/CALIBRATION_REPORT.md", report.encode())
    manifest_sha = write_manifest()
    status = {
        "schema": "safeconf_e174_joint_calibration_status_v1",
        "status": "COMPLETE",
        "gate_commit": gate_commit,
        "n_calibration_targets": 160,
        "n_calibration_tasks": 480,
        "calibration_targeting_x_values_read": 960,
        "evaluation_targeting_x_values_read": 0,
        "selected_model_spec": method["selected_model_spec"],
        "pair_mean_bound_violations": 0,
        "pair_max_bound_violations": 0,
        "squared_error_decomposition_max_abs_residual": max_identity_residual,
        "manifest_sha256": manifest_sha,
        "deployment_authorized": False,
        "wall_seconds": time.time() - started,
    }
    atomic_json(STAGING / "RUN_STATUS.json", status)
    os.replace(STAGING, OUT)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-commit", required=True)
    parser.add_argument("--branch", required=True)
    args = parser.parse_args()
    result = run(args.gate_commit, args.branch)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
