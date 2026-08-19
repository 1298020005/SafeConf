#!/usr/bin/env python3
"""Build physically isolated E174 pretruth, calibration, or evaluation assets."""

from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = ROOT / "tools/scripts/build_e168_primary_cd4_isolated_assets.py"
EXPERIMENT_REL = Path("docs/实验结果/E174_rotated_donor_conformal_certificate_20260719")
EXPERIMENT = ROOT / EXPERIMENT_REL
DATA_ROOT = Path("/home/yyf/data/safeconf_external/primary_cd4_perturbseq_2025")
FROZEN_METADATA_COMMIT = "0508a3380bc98bb3a7d0e4fe938cb9112c8187d7"
PANELS = ("R01", "R02", "R03", "R04")
WRAPPER_REL = Path("tools/scripts/build_e174_rotated_donor_panel_assets.py")
FREEZE_REL = Path("tools/scripts/freeze_e174_rotated_donor_conformal_certificate.py")
EXPECTED_PHASE_COUNTS = {
    "PRETRUTH_CONTROL_X": 11018,
    "PRETRUTH_TRAIN_X": 1920,
    "PRETRUTH_VALIDATION_X": 960,
    "POSTGATE_CALIBRATION_TRUTH_X": 240,
    "POSTCALIBRATION_EVALUATION_TRUTH_X": 960,
    "FORBIDDEN_COLUMN_UNSEEN_X": 720,
}


def import_helper() -> Any:
    spec = importlib.util.spec_from_file_location("safeconf_e174_asset_helper", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import asset helper: {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def configure(helper: Any, panel: str) -> Any:
    if panel not in PANELS:
        raise ValueError(f"unknown E174 panel: {panel}")
    panel_manifest = EXPERIMENT_REL / "manifests" / panel
    helper.EXPERIMENT_ID = f"E174_rotated_donor_conformal_certificate::{panel}"
    helper.TASK_PREFIX = f"E174::{panel}"
    helper.PRETRUTH_ASSET_STAGE = f"E174_{panel}_F2_PRETRUTH_ISOLATED_ASSET_BUILD"
    helper.POSTGATE_ASSET_STAGE = f"E174_{panel}_F3A_CALIBRATION_TRUTH_BUILD"
    helper.PRETRUTH_GATE_STAGE = f"E174_{panel}_F2_PRETRUTH_GATE"
    helper.EXPERIMENT_REL = EXPERIMENT_REL
    helper.EXPERIMENT = EXPERIMENT
    helper.FROZEN_METADATA_COMMIT = FROZEN_METADATA_COMMIT
    helper.SOURCE_LOCK_REL = EXPERIMENT_REL / "SOURCE_LOCK.json"
    helper.RUN_STATUS_REL = EXPERIMENT_REL / "RUN_STATUS.json"
    helper.MODEL_LOCK_REL = EXPERIMENT_REL / "MODEL_INPUT_LOCK.json"
    helper.STAT_LOCK_REL = EXPERIMENT_REL / "STATISTICAL_ANALYSIS_LOCK.json"
    helper.PLAN_REL = EXPERIMENT_REL / "PREREG_ANALYSIS_PLAN.md"
    helper.DONOR_ROLES_REL = EXPERIMENT_REL / "manifests/E174_DONOR_STATE_ROLES.csv"
    helper.ROW_ACCESS_REL = panel_manifest / f"E174_{panel}_ROW_ACCESS_MANIFEST.csv"
    helper.TARGETS_REL = panel_manifest / f"E174_{panel}_SELECTED_TARGETS.csv"
    helper.TASKS_REL = panel_manifest / f"E174_{panel}_TASK_MANIFEST.csv"
    helper.FROZEN_INPUTS = (
        helper.SOURCE_LOCK_REL,
        helper.RUN_STATUS_REL,
        helper.MODEL_LOCK_REL,
        helper.STAT_LOCK_REL,
        helper.PLAN_REL,
        helper.DONOR_ROLES_REL,
        helper.ROW_ACCESS_REL,
        helper.TARGETS_REL,
        helper.TASKS_REL,
        FREEZE_REL,
    )
    helper.DATA_ROOT = DATA_ROOT
    helper.BYTE_ATTESTATION = DATA_ROOT / "E168_SOURCE_BYTE_ATTESTATION.json"
    helper.ISOLATED_ROOT = DATA_ROOT / "isolated/E174" / panel
    helper.F2_DIR = helper.ISOLATED_ROOT / "F2_pretruth"
    helper.F3_DIR = helper.ISOLATED_ROOT / "F3A_calibration"
    helper.BUILDER_REL = WRAPPER_REL
    helper.EXPECTED_PHASE_COUNTS = EXPECTED_PHASE_COUNTS

    original_verify_state = helper.verify_frozen_state
    original_validate_manifests = helper.validate_manifests

    def verify_e174_state() -> Any:
        frozen = original_verify_state()
        source_lock = {
            **frozen.source_lock,
            "local_target_path": frozen.source_lock["source_path"],
            "content_length_bytes": frozen.source_lock["source_bytes"],
            "checksum_crc64nvme_base64": frozen.source_lock["official_crc64nvme_base64"],
        }
        return replace(frozen, source_lock=source_lock)

    def validate_e174_manifests() -> tuple[Any, Any, Any, Any]:
        rows, targets, tasks, roles = original_validate_manifests()
        if "panel_id" not in targets or set(targets.panel_id.astype(str)) != {panel}:
            raise helper.IntegrityError(f"{panel} target manifest panel identity changed")
        if "panel_target_rank" not in targets:
            raise helper.IntegrityError(f"{panel} target rank is missing")
        targets = targets.copy()
        targets["target_selection_rank"] = targets.panel_target_rank.astype(int)
        if targets.target_selection_rank.tolist() != list(range(1, 201)):
            raise helper.IntegrityError(f"{panel} target ranks are not 1..200")
        if set(tasks.panel_id.astype(str)) != {panel}:
            raise helper.IntegrityError(f"{panel} task manifest panel identity changed")
        expected_partition = {"CALIBRATION_20PCT": 40, "EVALUATION_80PCT": 160}
        if targets.heldout_donor_partition.value_counts().to_dict() != expected_partition:
            raise helper.IntegrityError(f"{panel} heldout donor partition changed")
        task_counts = {
            "calibration": int(tasks.calibration_test_task.astype(str).str.lower().eq("true").sum()),
            "evaluation": int(tasks.evaluation_test_task.astype(str).str.lower().eq("true").sum()),
        }
        if task_counts != {"calibration": 120, "evaluation": 480}:
            raise helper.IntegrityError(f"{panel} truth task counts changed: {task_counts}")
        return rows, targets, tasks, roles

    helper.verify_frozen_state = verify_e174_state
    helper.validate_manifests = validate_e174_manifests
    return helper


def truth_flag(tasks: pd.DataFrame, column: str) -> pd.Series:
    values = tasks[column]
    if values.dtype == bool:
        return values
    return values.astype(str).str.lower().eq("true")


def verify_calibration_snapshot(
    helper: Any,
    snapshot_path: Path,
    gate_commit: str,
    branch: str,
    panel: str,
    source_sha256: str,
    f2_manifest_sha: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    try:
        relative = snapshot_path.resolve().relative_to(ROOT.resolve())
    except ValueError as exc:
        raise helper.IntegrityError("Calibration snapshot must be inside the repository") from exc
    try:
        committed = helper.git("show", f"{gate_commit}:{relative.as_posix()}").stdout
    except subprocess.CalledProcessError as exc:
        raise helper.IntegrityError("Calibration snapshot is absent from gate commit") from exc
    local = snapshot_path.read_bytes()
    if local != committed:
        raise helper.IntegrityError("Local calibration snapshot differs from committed bytes")
    if helper.git("merge-base", "--is-ancestor", gate_commit, "HEAD", check=False).returncode:
        raise helper.IntegrityError("Current HEAD does not contain calibration gate commit")

    remote_heads: dict[str, str] = {}
    for remote in ("origin", "github"):
        fetched = f"refs/remotes/{remote}/{branch}"
        result = helper.git(
            "fetch",
            "--quiet",
            remote,
            f"refs/heads/{branch}:{fetched}",
            check=False,
        )
        if result.returncode:
            raise helper.IntegrityError(f"could not verify calibration gate on {remote}")
        remote_head = helper.git_text("rev-parse", fetched)
        if helper.git(
            "merge-base", "--is-ancestor", gate_commit, remote_head, check=False
        ).returncode:
            raise helper.IntegrityError(f"{remote} does not contain calibration gate commit")
        remote_heads[remote] = remote_head

    snapshot = json.loads(local)
    required = {
        "schema": "safeconf_e174_joint_calibration_snapshot_v1",
        "experiment": "E174_rotated_donor_conformal_certificate",
        "stage": "F3A_CONFORMAL_CALIBRATION_GATE",
        "status": "PASS",
        "n_calibration_targets": 160,
        "n_calibration_tasks": 480,
        "evaluation_targeting_x_values_read": 0,
        "method_frozen_before_evaluation_truth": True,
    }
    mismatches = {
        key: {"expected": value, "observed": snapshot.get(key)}
        for key, value in required.items()
        if snapshot.get(key) != value
    }
    if mismatches:
        raise helper.IntegrityError(f"calibration gate is not an exact PASS: {mismatches}")
    if snapshot.get("source_full_sha256") != source_sha256:
        raise helper.IntegrityError("calibration gate source SHA changed")
    if snapshot.get("f2_manifest_sha256", {}).get(panel) != f2_manifest_sha:
        raise helper.IntegrityError(f"calibration gate does not bind {panel} F2")
    f3 = helper.ISOLATED_ROOT / "F3A_calibration"
    f3_manifest_sha = helper.verify_manifest(f3)
    if snapshot.get("f3_calibration_manifest_sha256", {}).get(panel) != f3_manifest_sha:
        raise helper.IntegrityError(f"calibration gate does not bind {panel} F3A")
    return snapshot, remote_heads


def build_truth_stage(
    helper: Any,
    panel: str,
    stage: str,
    batch_size: int,
    snapshot_path: Path,
    gate_commit: str,
    branch: str,
) -> Path:
    if stage not in {"calibration", "evaluation"}:
        raise ValueError(stage)
    frozen = helper.verify_frozen_state()
    rows, targets, tasks, _roles = helper.validate_manifests()
    source, source_sha256 = helper.verify_complete_source(frozen)
    f2_manifest_sha = helper.verify_manifest(helper.F2_DIR)
    f2_attestation = json.loads((helper.F2_DIR / "ACCESS_ATTESTATION.json").read_text())
    if (
        f2_attestation.get("status") != "PASS"
        or f2_attestation.get("source_full_sha256") != source_sha256
    ):
        raise helper.IntegrityError("F2 attestation changed")

    if stage == "calibration":
        snapshot, remote_heads = helper.verify_gate_snapshot(
            snapshot_path, gate_commit, branch
        )
        if snapshot.get("source_full_sha256") != source_sha256:
            raise helper.IntegrityError("pretruth gate source SHA changed")
        if snapshot.get("f2_manifest_sha256") != f2_manifest_sha:
            raise helper.IntegrityError("pretruth gate does not bind F2")
        destination = helper.ISOLATED_ROOT / "F3A_calibration"
        phase = "POSTGATE_CALIBRATION_TRUTH_X"
        flag = "calibration_test_task"
        expected_tasks, expected_guides = 120, 240
        prefix = "CALIBRATION"
        asset_stage = f"E174_{panel}_F3A_CALIBRATION_TRUTH_BUILD"
        upstream_manifest = {"f2_manifest_sha256": f2_manifest_sha}
    else:
        snapshot, remote_heads = verify_calibration_snapshot(
            helper,
            snapshot_path,
            gate_commit,
            branch,
            panel,
            source_sha256,
            f2_manifest_sha,
        )
        destination = helper.ISOLATED_ROOT / "F4_evaluation"
        phase = "POSTCALIBRATION_EVALUATION_TRUTH_X"
        flag = "evaluation_test_task"
        expected_tasks, expected_guides = 480, 960
        prefix = "EVALUATION"
        asset_stage = f"E174_{panel}_F4_EVALUATION_TRUTH_BUILD"
        upstream_manifest = {
            "f2_manifest_sha256": f2_manifest_sha,
            "f3_calibration_manifest_sha256": helper.verify_manifest(
                helper.ISOLATED_ROOT / "F3A_calibration"
            ),
        }

    staging = helper.prepare_staging(destination)
    try:
        gene_panel = pd.read_csv(helper.F2_DIR / "GENE_PANEL.csv", keep_default_na=False)
        if len(gene_panel) != helper.PANEL_SIZE or gene_panel.panel_index.tolist() != list(
            range(helper.PANEL_SIZE)
        ):
            raise helper.IntegrityError("F2 panel schema/order changed")
        panel_columns = gene_panel.source_column_index.to_numpy(dtype=np.int64)
        with np.load(helper.F2_DIR / "CONTROL_PROFILES.npz", allow_pickle=False) as asset:
            controls = {key: np.asarray(asset[key], dtype=np.float64) for key in asset.files}
        selected_rows = rows.loc[rows.x_access_phase.eq(phase)].copy()
        with h5py.File(source, "r") as handle:
            reader = helper.RowMatrixReader(handle)
            effects, guide_effects, access_rows, guide_table = helper.consume_target_rows(
                reader=reader,
                selected_rows=selected_rows,
                panel_columns=panel_columns,
                controls=controls,
                expected_guides=helper.expected_guides_by_target(targets),
                valid_task_ids=set(tasks.task_id.astype(str)),
                stage=f"E174_{stage.upper()}",
                batch_size=batch_size,
            )
        if len(effects) != expected_tasks or len(guide_effects) != expected_guides:
            raise helper.IntegrityError(
                f"{stage} truth counts changed: tasks={len(effects)}, guides={len(guide_effects)}"
            )
        if set(guide_table.x_access_phase.astype(str)) != {phase}:
            raise helper.IntegrityError(f"non-{stage} X entered isolated asset")

        helper.save_npz(staging / f"{prefix}_TARGET_EFFECTS.npz", effects)
        helper.save_npz(staging / f"{prefix}_GUIDE_EFFECTS.npz", guide_effects)
        guide_table.to_csv(staging / f"{prefix}_GUIDE_EFFECT_INDEX.csv", index=False)
        truth_tasks = tasks.loc[truth_flag(tasks, flag)].copy()
        if len(truth_tasks) != expected_tasks or set(truth_tasks.task_id) != set(effects):
            raise helper.IntegrityError(f"frozen {stage} tasks do not match truth keys")
        truth_tasks["truth_effect_asset_key"] = truth_tasks.task_id
        truth_tasks.to_csv(staging / f"{prefix}_TASKS.csv", index=False)
        access = pd.DataFrame(access_rows).sort_values("metadata_row_index", kind="stable")
        phase_counts = helper.assert_exact_access(access, rows, (phase,))
        access.to_csv(staging / "ROW_ACCESS_AUDIT.csv", index=False)

        primary_hashes = {
            path.name: helper.sha256_file(path) for path in staging.iterdir() if path.is_file()
        }
        attestation = {
            "experiment": helper.EXPERIMENT_ID,
            "stage": asset_stage,
            "status": "PASS",
            "deployment_authorized": False,
            "frozen_metadata_commit": FROZEN_METADATA_COMMIT,
            "current_git_head": frozen.current_head,
            "code_freeze_branch": frozen.branch,
            "code_freeze_remote_heads": frozen.remote_heads,
            "builder_sha256": frozen.builder_sha256,
            "source_path": str(source),
            "source_bytes": source.stat().st_size,
            "source_full_sha256": source_sha256,
            "source_official_crc64nvme_base64": frozen.source_lock[
                "checksum_crc64nvme_base64"
            ],
            "source_full_sha256_recomputed_before_x_access": True,
            "gate_snapshot_path": str(snapshot_path.resolve().relative_to(ROOT.resolve())),
            "gate_snapshot_sha256": helper.sha256_file(snapshot_path),
            "gate_commit": gate_commit,
            "gate_remote_heads": remote_heads,
            **upstream_manifest,
            "logical_x_rows_read": len(access),
            "logical_x_rows_read_by_phase": phase_counts,
            "all_returned_x_rows_read_exactly_once": True,
            "calibration_targeting_x_values_read": len(access) if stage == "calibration" else 0,
            "evaluation_targeting_x_values_read": len(access) if stage == "evaluation" else 0,
            "other_truth_partition_x_values_read": 0,
            "train_or_validation_targeting_x_values_read_in_truth_stage": 0,
            f"n_{stage}_target_effects": len(effects),
            f"n_{stage}_guide_effects": len(guide_effects),
            "test_performance_metrics_computed": 0,
            "primary_output_sha256": primary_hashes,
        }
        helper.atomic_write_text(
            staging / "ACCESS_ATTESTATION.json",
            json.dumps(attestation, indent=2, sort_keys=True) + "\n",
        )
        helper.write_manifest(staging)
        helper.complete_staging(staging, destination)
        return destination
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=PANELS, required=True)
    parser.add_argument("--stage", choices=("pretruth", "calibration", "evaluation"), required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gate-snapshot", type=Path)
    parser.add_argument("--gate-commit")
    parser.add_argument("--branch")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    helper = configure(import_helper(), args.panel)
    if args.self_test:
        helper.self_test()
        print(json.dumps({"status": "PASS", "test": "E174_asset_wrapper+E168_primitives"}))
        return
    if args.stage == "pretruth":
        output = helper.build_pretruth(args.batch_size)
    else:
        if not args.gate_snapshot or not args.gate_commit or not args.branch:
            parser.error(f"{args.stage} requires gate snapshot, commit, and branch")
        output = build_truth_stage(
            helper,
            args.panel,
            args.stage,
            args.batch_size,
            args.gate_snapshot,
            args.gate_commit,
            args.branch,
        )
    print(
        json.dumps(
            {"status": "PASS", "panel": args.panel, "stage": args.stage, "output": str(output)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
