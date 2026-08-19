#!/usr/bin/env python3
"""Open only the E176 calibration or evaluation rows authorized by committed gates."""

from __future__ import annotations

import argparse
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
CONFIG_PATH = ROOT / "tools/scripts/build_e176_four_donor_panel_assets.py"
DATA_ROOT = Path("/home/yyf/data/safeconf_external/primary_cd4_perturbseq_2025")
EXPERIMENT = ROOT / "docs/实验结果/E176_four_donor_fresh_confirmation_20260719"
PANELS = ("H01", "H02", "H03", "H04")
BUILDER_REL = Path("tools/scripts/build_e176_truth_assets.py")


def import_config() -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location("safeconf_e176_truth_config", CONFIG_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import E176 asset configuration: {CONFIG_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )


def git_text(*args: str) -> str:
    return git(*args).stdout.decode().strip()


def strict_flag(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    values = series.astype(str).str.lower()
    if not values.isin({"true", "false"}).all():
        raise RuntimeError("non-boolean frozen task flag")
    return values.eq("true")


def verify_snapshot(
    helper: Any,
    path: Path,
    code_freeze_commit: str,
    branch: str,
    stage: str,
    panel: str,
    source_sha: str,
    f2_manifest: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    if git("merge-base", "--is-ancestor", code_freeze_commit, "HEAD", check=False).returncode:
        raise helper.IntegrityError("current HEAD does not contain truth-stage code freeze")
    relative = path.relative_to(ROOT).as_posix()
    try:
        committed = git("show", f"{code_freeze_commit}:{relative}").stdout
    except subprocess.CalledProcessError as exc:
        raise helper.IntegrityError(f"gate snapshot absent from code freeze: {relative}") from exc
    if committed != path.read_bytes():
        raise helper.IntegrityError("local gate snapshot differs from committed bytes")
    remote_heads: dict[str, str] = {}
    for remote in ("origin", "github"):
        fetched = f"refs/remotes/{remote}/{branch}"
        result = git("fetch", "--quiet", remote, f"refs/heads/{branch}:{fetched}", check=False)
        if result.returncode:
            raise helper.IntegrityError(f"cannot verify truth gate on {remote}")
        remote_head = git_text("rev-parse", fetched)
        if git("merge-base", "--is-ancestor", code_freeze_commit, remote_head, check=False).returncode:
            raise helper.IntegrityError(f"code freeze absent from {remote}/{branch}")
        remote_heads[remote] = remote_head
    snapshot = json.loads(committed)
    if stage == "calibration":
        required = {
            "schema": "safeconf_e176_joint_pretruth_gate_v1",
            "status": "PASS",
            "decision": "CALIBRATION_TRUTH_ACCESS_AUTHORIZED",
            "g4_units": 24,
            "g4_units_passed": 24,
            "test_donor_targeting_x_values_read": 0,
            "calibration_targeting_x_values_read": 0,
            "evaluation_targeting_x_values_read": 0,
            "calibration_runner_authorized": True,
            "final_evaluator_authorized": False,
        }
        expected_f2 = snapshot.get("panel_evidence", {}).get(panel, {}).get(
            "f2_manifest_sha256"
        )
        if expected_f2 != f2_manifest:
            raise helper.IntegrityError(f"{panel} F2 is not bound by joint pretruth gate")
    else:
        required = {
            "schema": "safeconf_e176_donor_specific_calibration_gate_v1",
            "status": "PASS",
            "n_calibration_targets": 160,
            "n_calibration_tasks": 480,
            "evaluation_targeting_x_values_read": 0,
            "method_frozen_before_evaluation_truth": True,
            "final_evaluator_authorized": True,
        }
        expected_f2 = snapshot.get("f2_manifest_sha256", {}).get(panel)
        if expected_f2 != f2_manifest:
            raise helper.IntegrityError(f"{panel} F2 is not bound by calibration gate")
        f3_manifest = helper.verify_manifest(
            DATA_ROOT / "isolated/E176" / panel / "F3_calibration"
        )
        if snapshot.get("f3_calibration_manifest_sha256", {}).get(panel) != f3_manifest:
            raise helper.IntegrityError(f"{panel} F3 is not bound by calibration gate")
    changed = {
        key: {"expected": value, "observed": snapshot.get(key)}
        for key, value in required.items() if snapshot.get(key) != value
    }
    if changed:
        raise helper.IntegrityError(f"{stage} gate changed: {changed}")
    if snapshot.get("source_full_sha256") != source_sha:
        raise helper.IntegrityError("truth gate source SHA changed")
    return snapshot, remote_heads


def build(
    panel: str,
    stage: str,
    batch_size: int,
    snapshot_path: Path,
    code_freeze_commit: str,
    branch: str,
) -> Path:
    config = import_config()
    helper = config.configure(config.import_helper(), panel)
    helper.BUILDER_REL = BUILDER_REL
    helper.F3_DIR = helper.ISOLATED_ROOT / "F3_calibration"
    frozen = helper.verify_frozen_state()
    rows, targets, tasks, _roles = helper.validate_manifests()
    source, source_sha = helper.verify_complete_source(frozen)
    f2 = helper.ISOLATED_ROOT / "F2_pretruth"
    f2_manifest = helper.verify_manifest(f2)
    f2_attestation = json.loads((f2 / "ACCESS_ATTESTATION.json").read_text())
    if (
        f2_attestation.get("status") != "PASS"
        or f2_attestation.get("source_full_sha256") != source_sha
    ):
        raise helper.IntegrityError("F2 attestation changed")
    snapshot, gate_remotes = verify_snapshot(
        helper, snapshot_path, code_freeze_commit, branch, stage, panel,
        source_sha, f2_manifest,
    )

    if stage == "calibration":
        destination = helper.ISOLATED_ROOT / "F3_calibration"
        phase = "POSTGATE_CALIBRATION_TRUTH_X"
        flag = "calibration_test_task"
        prefix = "CALIBRATION"
        expected_tasks, expected_guides = 120, 240
        asset_stage = f"E176_{panel}_F3_CALIBRATION_TRUTH_BUILD"
        upstream = {"f2_manifest_sha256": f2_manifest}
    else:
        destination = helper.ISOLATED_ROOT / "F4_evaluation"
        phase = "POSTCALIBRATION_EVALUATION_TRUTH_X"
        flag = "evaluation_test_task"
        prefix = "EVALUATION"
        expected_tasks, expected_guides = 480, 960
        asset_stage = f"E176_{panel}_F4_EVALUATION_TRUTH_BUILD"
        upstream = {
            "f2_manifest_sha256": f2_manifest,
            "f3_calibration_manifest_sha256": helper.verify_manifest(
                helper.ISOLATED_ROOT / "F3_calibration"
            ),
        }

    staging = helper.prepare_staging(destination)
    try:
        gene_panel = pd.read_csv(f2 / "GENE_PANEL.csv", keep_default_na=False)
        if len(gene_panel) != helper.PANEL_SIZE:
            raise helper.IntegrityError("F2 gene panel changed")
        panel_columns = gene_panel.source_column_index.to_numpy(dtype=np.int64)
        with np.load(f2 / "CONTROL_PROFILES.npz", allow_pickle=False) as archive:
            controls = {key: np.asarray(archive[key], dtype=np.float64) for key in archive.files}
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
                stage=f"E176_{stage.upper()}",
                batch_size=batch_size,
            )
        if len(effects) != expected_tasks or len(guide_effects) != expected_guides:
            raise helper.IntegrityError(
                f"{stage} truth counts changed: {len(effects)}/{len(guide_effects)}"
            )
        if set(guide_table.x_access_phase.astype(str)) != {phase}:
            raise helper.IntegrityError(f"non-{stage} X entered isolated asset")
        helper.save_npz(staging / f"{prefix}_TARGET_EFFECTS.npz", effects)
        helper.save_npz(staging / f"{prefix}_GUIDE_EFFECTS.npz", guide_effects)
        guide_table.to_csv(staging / f"{prefix}_GUIDE_EFFECT_INDEX.csv", index=False)
        truth_tasks = tasks.loc[strict_flag(tasks[flag])].copy()
        if len(truth_tasks) != expected_tasks or set(truth_tasks.task_id) != set(effects):
            raise helper.IntegrityError(f"{stage} task keys changed")
        truth_tasks["truth_effect_asset_key"] = truth_tasks.task_id
        truth_tasks.to_csv(staging / f"{prefix}_TASKS.csv", index=False)
        access = pd.DataFrame(access_rows).sort_values("metadata_row_index", kind="stable")
        phase_counts = helper.assert_exact_access(access, rows, (phase,))
        access.to_csv(staging / "ROW_ACCESS_AUDIT.csv", index=False)
        primary_hashes = {
            path.name: helper.sha256_file(path) for path in staging.iterdir() if path.is_file()
        }
        attestation = {
            "experiment": f"E176_four_donor_fresh_confirmation::{panel}",
            "stage": asset_stage,
            "status": "PASS",
            "deployment_authorized": False,
            "frozen_metadata_commit": config.FROZEN_METADATA_COMMIT,
            "current_git_head": frozen.current_head,
            "code_freeze_branch": frozen.branch,
            "code_freeze_remote_heads": frozen.remote_heads,
            "builder_sha256": frozen.builder_sha256,
            "source_path": str(source),
            "source_bytes": source.stat().st_size,
            "source_full_sha256": source_sha,
            "source_full_sha256_recomputed_before_x_access": True,
            "gate_snapshot_path": snapshot_path.resolve().relative_to(ROOT.resolve()).as_posix(),
            "gate_snapshot_sha256": helper.sha256_file(snapshot_path),
            "gate_commit": code_freeze_commit,
            "gate_remote_heads": gate_remotes,
            **upstream,
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
    parser.add_argument("--stage", choices=("calibration", "evaluation"), required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gate-snapshot", type=Path, required=True)
    parser.add_argument("--code-freeze-commit", required=True)
    parser.add_argument("--branch", required=True)
    args = parser.parse_args()
    output = build(
        args.panel, args.stage, args.batch_size, args.gate_snapshot,
        args.code_freeze_commit, args.branch,
    )
    print(json.dumps({
        "status": "PASS", "panel": args.panel, "stage": args.stage,
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
