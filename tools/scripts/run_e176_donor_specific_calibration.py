#!/usr/bin/env python3
"""Calibrate E176 donor-specific risk upper bounds while evaluation truth stays sealed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
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
EXPERIMENT = ROOT / "docs/实验结果/E176_four_donor_fresh_confirmation_20260719"
DATA_ROOT = Path("/home/yyf/data/safeconf_external/primary_cd4_perturbseq_2025")
OUT = EXPERIMENT / "calibration_release"
STAGING = EXPERIMENT / ".calibration_release.staging"
JOINT_GATE = EXPERIMENT / "pretruth_joint/PRETRUTH_GATE_SNAPSHOT.json"
METHOD_ROOT = ROOT / "docs/实验结果/E174_rotated_donor_conformal_certificate_20260719/method_development"
METHOD_SNAPSHOT = METHOD_ROOT / "METHOD_GATE_SNAPSHOT.json"
METHOD_STATUS = METHOD_ROOT / "RUN_STATUS.json"
METHOD_MANIFEST = METHOD_ROOT / "MANIFEST.sha256"
PANELS = ("H01", "H02", "H03", "H04")
SCRIPT = Path(__file__).resolve()
COMMON = ROOT / "tools/scripts/e174_conformal_common.py"
TRUTH_BUILDER = ROOT / "tools/scripts/build_e176_truth_assets.py"
PRETRUTH_BUILDER = ROOT / "tools/scripts/build_e176_four_donor_panel_assets.py"
PRETRUTH_RUNNER = ROOT / "tools/scripts/run_e176_four_donor_panel_pretruth.py"
JOINT_GATE_SCRIPT = ROOT / "tools/scripts/freeze_e176_joint_pretruth_gate.py"
FREEZER = ROOT / "tools/scripts/freeze_e176_four_donor_fresh_confirmation.py"
FINAL_EVALUATOR = ROOT / "tools/scripts/run_e176_final_evaluation.py"


class IntegrityFailure(RuntimeError):
    pass


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


def require_committed(path: Path, commit: str) -> dict[str, Any]:
    relative = path.relative_to(ROOT).as_posix()
    try:
        committed = git("show", f"{commit}:{relative}").stdout
    except subprocess.CalledProcessError as exc:
        raise IntegrityFailure(f"file absent from calibration code freeze: {relative}") from exc
    if committed != path.read_bytes():
        raise IntegrityFailure(f"file changed after calibration code freeze: {relative}")
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def verify_code_freeze(commit: str, branch: str) -> tuple[dict[str, str], list[dict[str, Any]]]:
    if git("merge-base", "--is-ancestor", commit, "HEAD", check=False).returncode:
        raise IntegrityFailure("current HEAD does not descend from calibration code freeze")
    files = [
        SCRIPT,
        COMMON,
        TRUTH_BUILDER,
        PRETRUTH_BUILDER,
        PRETRUTH_RUNNER,
        JOINT_GATE_SCRIPT,
        FREEZER,
        FINAL_EVALUATOR,
        JOINT_GATE,
        EXPERIMENT / "RUN_STATUS.json",
        EXPERIMENT / "MODEL_INPUT_LOCK.json",
        EXPERIMENT / "STATISTICAL_ANALYSIS_LOCK.json",
        EXPERIMENT / "PREREG_ANALYSIS_PLAN.md",
        EXPERIMENT / "manifests/E176_ALL_SELECTED_TARGETS.csv",
        EXPERIMENT / "manifests/E176_ALL_TASKS.csv",
        METHOD_SNAPSHOT,
        METHOD_STATUS,
        METHOD_MANIFEST,
    ]
    hashes = [require_committed(path, commit) for path in files]
    remote_heads: dict[str, str] = {}
    for remote in ("origin", "github"):
        fetched = f"refs/remotes/{remote}/{branch}"
        result = git("fetch", "--quiet", remote, f"refs/heads/{branch}:{fetched}", check=False)
        if result.returncode:
            raise IntegrityFailure(f"cannot verify calibration freeze on {remote}")
        remote_head = git_text("rev-parse", fetched)
        if git("merge-base", "--is-ancestor", commit, remote_head, check=False).returncode:
            raise IntegrityFailure(f"calibration freeze absent from {remote}/{branch}")
        remote_heads[remote] = remote_head
    gate = json.loads(JOINT_GATE.read_text())
    required = {
        "schema": "safeconf_e176_joint_pretruth_gate_v1",
        "status": "PASS",
        "decision": "CALIBRATION_TRUTH_ACCESS_AUTHORIZED",
        "g4_units": 24,
        "g4_units_passed": 24,
        "calibration_targeting_x_values_read": 0,
        "evaluation_targeting_x_values_read": 0,
    }
    if any(gate.get(key) != value for key, value in required.items()):
        raise IntegrityFailure("joint pretruth gate changed")
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


def load_panel(panel: str, commit: str) -> tuple[pd.DataFrame, dict[str, str], list[dict[str, Any]]]:
    release = EXPERIMENT / "pretruth_release" / panel
    snapshot_path = release / "PRETRUTH_GATE_SNAPSHOT.json"
    scoring_path = release / "tables/PRETRUTH_SCORING_INTERFACE.csv"
    prediction_path = release / "arrays/PRETRUTH_PREDICTIONS.npz"
    hashes = [
        require_committed(snapshot_path, commit),
        require_committed(scoring_path, commit),
        require_committed(prediction_path, commit),
    ]
    snapshot = json.loads(snapshot_path.read_text())
    required = {
        "schema": f"safeconf_e176_{panel.lower()}_pretruth_gate_snapshot_v1",
        "status": "PASS",
        "test_targeting_x_values_read": 0,
        "test_query_graphs_containing_y": 0,
    }
    if any(snapshot.get(key) != value for key, value in required.items()):
        raise IntegrityFailure(f"{panel} pretruth snapshot changed")
    isolated = DATA_ROOT / "isolated/E176" / panel
    f2, f3 = isolated / "F2_pretruth", isolated / "F3_calibration"
    f2_manifest, f3_manifest = sha256_file(f2 / "MANIFEST.sha256"), sha256_file(
        f3 / "MANIFEST.sha256"
    )
    if snapshot.get("f2_manifest_sha256") != f2_manifest:
        raise IntegrityFailure(f"{panel} F2 manifest changed")
    f2_att = json.loads((f2 / "ACCESS_ATTESTATION.json").read_text())
    f3_att = json.loads((f3 / "ACCESS_ATTESTATION.json").read_text())
    required_f3 = {
        "experiment": f"E176_four_donor_fresh_confirmation::{panel}",
        "stage": f"E176_{panel}_F3_CALIBRATION_TRUTH_BUILD",
        "status": "PASS",
        "f2_manifest_sha256": f2_manifest,
        "source_full_sha256": snapshot["source_full_sha256"],
        "gate_commit": commit,
        "calibration_targeting_x_values_read": 240,
        "evaluation_targeting_x_values_read": 0,
        "other_truth_partition_x_values_read": 0,
        "n_calibration_target_effects": 120,
        "n_calibration_guide_effects": 240,
        "test_performance_metrics_computed": 0,
    }
    changed = {
        key: {"expected": value, "observed": f3_att.get(key)}
        for key, value in required_f3.items() if f3_att.get(key) != value
    }
    if changed:
        raise IntegrityFailure(f"{panel} F3 attestation failed: {changed}")
    if f2_att.get("source_full_sha256") != f3_att.get("source_full_sha256"):
        raise IntegrityFailure(f"{panel} F2/F3 source changed")
    if f2_att.get("builder_sha256") != sha256_file(PRETRUTH_BUILDER):
        raise IntegrityFailure(f"{panel} F2 builder changed")
    if f3_att.get("builder_sha256") != sha256_file(TRUTH_BUILDER):
        raise IntegrityFailure(f"{panel} F3 builder changed")

    scoring = pd.read_csv(scoring_path, keep_default_na=False)
    if len(scoring) != 2160 or scoring.task_id.nunique() != 2160:
        raise IntegrityFailure(f"{panel} scoring interface changed")
    locked = pd.read_csv(
        EXPERIMENT / f"manifests/{panel}/E176_{panel}_TASK_MANIFEST.csv",
        keep_default_na=False,
    )
    calibration_tasks = locked.loc[strict_flag(locked.calibration_test_task)].copy()
    ids = calibration_tasks.task_id.astype(str).tolist()
    if len(ids) != 120:
        raise IntegrityFailure(f"{panel} calibration task count changed")
    score_index = scoring.set_index("task_id")
    metadata = score_index.loc[ids].reset_index()
    if metadata.heldout_donor_partition.astype(str).ne("CALIBRATION_20PCT").any():
        raise IntegrityFailure(f"{panel} evaluation task entered calibration")
    with np.load(prediction_path, allow_pickle=False) as archive:
        row = {task: index for index, task in enumerate(scoring.task_id.astype(str))}
        predictions = {
            name: np.stack([np.asarray(archive[name][row[task]], dtype=float) for task in ids])
            for name in ("scGPT_seed_mean", "GEARS_seed_mean", "ensemble_seed_family_mean")
        }
        required_seed_keys = {
            f"{family}_seed{seed}" for family in ("scGPT", "GEARS")
            for seed in (3407, 3408, 3409, 3410, 3411)
        }
        if not required_seed_keys.issubset(set(archive.files)):
            raise IntegrityFailure(f"{panel} five-seed predictions are incomplete")
    truth = load_npz_vectors(f3 / "CALIBRATION_TARGET_EFFECTS.npz", 120)
    if set(truth) != set(ids):
        raise IntegrityFailure(f"{panel} calibration truth/query keys differ")
    truth_matrix = np.stack([truth[task] for task in ids])
    sc, ge, ensemble = (
        predictions["scGPT_seed_mean"],
        predictions["GEARS_seed_mean"],
        predictions["ensemble_seed_family_mean"],
    )
    disagreement = rmse_rows(sc, ge)
    if not np.allclose(
        disagreement, metadata.model_disagreement_rmse.to_numpy(float),
        atol=2e-7, rtol=2e-6,
    ):
        raise IntegrityFailure(f"{panel} frozen disagreement changed")
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
    hashes.extend([
        {"path": str(f2 / "MANIFEST.sha256"), "bytes": (f2 / "MANIFEST.sha256").stat().st_size,
         "sha256": f2_manifest},
        {"path": str(f3 / "MANIFEST.sha256"), "bytes": (f3 / "MANIFEST.sha256").stat().st_size,
         "sha256": f3_manifest},
    ])
    return metrics, {
        "source_full_sha256": snapshot["source_full_sha256"],
        "f2_manifest_sha256": f2_manifest,
        "f3_manifest_sha256": f3_manifest,
    }, hashes


def flatten_rules(rules: dict[str, dict[str, dict[str, dict[str, Any]]]]) -> pd.DataFrame:
    rows = []
    for outcome, by_panel in rules.items():
        for panel, by_spec in by_panel.items():
            for spec, rule in by_spec.items():
                rows.append({"outcome": outcome, "panel_id": panel, "model_spec": spec, **rule})
    return pd.DataFrame(rows)


def write_manifest() -> str:
    files = sorted(path for path in STAGING.rglob("*")
                   if path.is_file() and path.name not in {"MANIFEST.sha256", "RUN_STATUS.json"})
    payload = "".join(
        f"{sha256_file(path)}  {path.relative_to(STAGING).as_posix()}\n" for path in files
    )
    atomic_bytes(STAGING / "MANIFEST.sha256", payload.encode())
    return sha256_file(STAGING / "MANIFEST.sha256")


def run(commit: str, branch: str) -> dict[str, Any]:
    started = time.time()
    if OUT.exists() or STAGING.exists():
        raise IntegrityFailure("E176 calibration output is append-only")
    remote_heads, code_hashes = verify_code_freeze(commit, branch)
    method = json.loads(METHOD_SNAPSHOT.read_text())
    method_status = json.loads(METHOD_STATUS.read_text())
    if method.get("selected_model_spec") != {
        "ensemble_rmse": "magnitude", "pair_mean_rmse": "magnitude"
    }:
        raise IntegrityFailure("frozen magnitude fallback changed")
    if sha256_file(METHOD_MANIFEST) != method_status.get("manifest_sha256"):
        raise IntegrityFailure("method development manifest changed")

    metrics_list, input_hashes, metadata = [], [], {}
    for panel in PANELS:
        frame, panel_meta, hashes = load_panel(panel, commit)
        metrics_list.append(frame)
        metadata[panel] = panel_meta
        input_hashes.extend(hashes)
    metrics = add_pair_columns(pd.concat(metrics_list, ignore_index=True))
    if len(metrics) != 480 or target_key(metrics).nunique() != 160:
        raise IntegrityFailure("calibration population must be 160 targets / 480 tasks")
    pair_mean_violations = int(
        (metrics.pair_lower_bound_rmse > metrics.pair_mean_rmse + 1e-10).sum()
    )
    pair_max_violations = int(
        (metrics.pair_lower_bound_rmse > metrics.pair_max_rmse + 1e-10).sum()
    )
    max_identity_residual = float(metrics.decomposition_abs_residual.max())
    if pair_mean_violations or pair_max_violations or max_identity_residual > 1e-8:
        raise IntegrityFailure("calibration certificate identity failed")

    rules: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for outcome in ("ensemble_rmse", "pair_mean_rmse"):
        rules[outcome] = {}
        for panel in PANELS:
            rules[outcome][panel] = {}
            mask = metrics.panel_id.eq(panel)
            if target_key(metrics.loc[mask]).nunique() != 40:
                raise IntegrityFailure(f"{panel} calibration target count changed")
            for spec in MODEL_SPECS:
                model = method["models"][outcome][spec]
                base = predict_ridge(metrics.loc[mask], model)
                rule = calibrate_cluster_upper(metrics.loc[mask], base, outcome, 0.90)
                if rule["finite_sample_order_rank_one_based"] != 37:
                    raise IntegrityFailure(f"{panel} conformal order rank changed")
                rules[outcome][panel][spec] = rule
                metrics.loc[mask, f"base_{outcome}__{spec}"] = base

    source_hashes = {value["source_full_sha256"] for value in metadata.values()}
    if len(source_hashes) != 1:
        raise IntegrityFailure("calibration panels use different source objects")
    for sub in ("tables", "reports"):
        (STAGING / sub).mkdir(parents=True, exist_ok=False)
    atomic_csv(STAGING / "tables/CALIBRATION_TASK_METRICS.csv", metrics)
    atomic_csv(STAGING / "tables/CALIBRATION_RULES.csv", flatten_rules(rules))
    atomic_csv(STAGING / "tables/INPUT_HASHES.csv", pd.DataFrame(code_hashes + input_hashes))
    snapshot = {
        "schema": "safeconf_e176_donor_specific_calibration_gate_v1",
        "experiment": "E176_four_donor_fresh_confirmation",
        "stage": "F3_DONOR_SPECIFIC_CONFORMAL_CALIBRATION",
        "status": "PASS",
        "calibration_code_freeze_commit": commit,
        "calibration_code_freeze_remote_heads": remote_heads,
        "source_full_sha256": next(iter(source_hashes)),
        "f2_manifest_sha256": {
            panel: metadata[panel]["f2_manifest_sha256"] for panel in PANELS
        },
        "f3_calibration_manifest_sha256": {
            panel: metadata[panel]["f3_manifest_sha256"] for panel in PANELS
        },
        "method_snapshot_sha256": sha256_file(METHOD_SNAPSHOT),
        "selected_model_spec": method["selected_model_spec"],
        "models": method["models"],
        "calibration_rules": rules,
        "n_calibration_targets_each_donor": 40,
        "finite_sample_order_rank_each_donor": 37,
        "n_calibration_targets": 160,
        "n_calibration_tasks": 480,
        "calibration_targeting_x_values_read": 960,
        "evaluation_targeting_x_values_read": 0,
        "method_frozen_before_evaluation_truth": True,
        "evaluation_truth_may_modify_model_or_quantile": False,
        "pair_mean_bound_violations": pair_mean_violations,
        "pair_max_bound_violations": pair_max_violations,
        "squared_error_decomposition_max_abs_residual": max_identity_residual,
        "final_evaluator_authorized": True,
        "deployment_authorized": False,
    }
    atomic_json(STAGING / "CALIBRATION_GATE_SNAPSHOT.json", snapshot)
    report = f"""# E176 donor-specific calibration

联合 pretruth gate 提交并通过后，只开放每位测试供体预分配的 40 个校准靶点，共 160 个靶点、480 个任务；640 个评价靶点 targeting X 读取数仍为 0。

每位供体单独计算 target-cluster residual 的第 37 顺序统计量。基础模型仍是 E174 在任何 E176 真值开放前冻结的 magnitude 规格；本阶段没有重选特征、阈值或供体。pair mean/max 下界违例均为 0，平方误差分解最大残差为 {max_identity_residual:.3g}。
"""
    atomic_bytes(STAGING / "reports/E176_CALIBRATION_REPORT.md", report.encode())
    manifest_sha = write_manifest()
    status = {
        "schema": "safeconf_e176_donor_specific_calibration_status_v1",
        "status": "COMPLETE",
        "code_freeze_commit": commit,
        "n_calibration_targets": 160,
        "n_calibration_tasks": 480,
        "calibration_targeting_x_values_read": 960,
        "evaluation_targeting_x_values_read": 0,
        "selected_model_spec": method["selected_model_spec"],
        "pair_mean_bound_violations": pair_mean_violations,
        "pair_max_bound_violations": pair_max_violations,
        "squared_error_decomposition_max_abs_residual": max_identity_residual,
        "manifest_sha256": manifest_sha,
        "deployment_authorized": False,
        "python": sys.version,
        "platform": platform.platform(),
        "wall_seconds": time.time() - started,
    }
    atomic_json(STAGING / "RUN_STATUS.json", status)
    os.replace(STAGING, OUT)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-freeze-commit", required=True)
    parser.add_argument("--branch", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.code_freeze_commit, args.branch), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
