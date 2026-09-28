#!/usr/bin/env python3
"""Build E170 test truth once, after committed SafeConf-v4 authorization.

The four old E170 panels are opened together.  The original aborted E170
directory is immutable; this builder writes a new v4-specific truth asset and
computes no prediction error or SafeConf metric.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "docs/实验结果/Stage2_mature_upstream_20260928"
AUTH = STAGE / "E170_V4_CONFIRMATION_AUTHORIZATION.json"
WRAPPER_PATH = ROOT / "tools/scripts/build_e170_primary_cd4_panel_assets.py"
HELPER_PATH = ROOT / "tools/scripts/build_e168_primary_cd4_isolated_assets.py"
PUBLIC = Path("/home/yyf/proj/docs/实验结果/E170_primary_cd4_multipanel_precision_20260718")
F2_ROOT = Path("/home/yyf/data/safeconf_external/primary_cd4_perturbseq_2025/isolated/E170")
OUTPUT = Path("/home/yyf/data/safeconf_v4_e170_confirmation_truth")
PANELS = ("P01", "P02", "P03", "P04")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], cwd=ROOT, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def import_wrapper():
    spec = importlib.util.spec_from_file_location("safeconf_e170_v4_wrapper", WRAPPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import the audited E170 asset wrapper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_authorization(commit: str, branch: str) -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite immutable truth asset: {OUTPUT}")
    relative = AUTH.relative_to(ROOT).as_posix()
    local = AUTH.read_bytes()
    committed = git("show", f"{commit}:{relative}").stdout
    if local != committed:
        raise RuntimeError("local authorization differs from committed bytes")
    if git("merge-base", "--is-ancestor", commit, "HEAD", check=False).returncode:
        raise RuntimeError("authorization commit is not an ancestor of HEAD")
    remote_ref = f"refs/remotes/github/{branch}"
    if git("show-ref", "--verify", "--quiet", remote_ref, check=False).returncode:
        raise RuntimeError(f"missing locally verified GitHub branch ref: {remote_ref}")
    if git("merge-base", "--is-ancestor", commit, remote_ref, check=False).returncode:
        raise RuntimeError("GitHub branch does not contain the authorization commit")
    auth = json.loads(local)
    required = {
        "status": "AUTHORIZED_AFTER_FINAL_CANDIDATE_FREEZE",
        "all_four_panels_required": True,
        "test_truth_opened_at_authorization": False,
        "n_test_tasks": 2400,
        "final_candidate": "V2_nested_evidence_shrinkage",
    }
    mismatches = {key: (value, auth.get(key)) for key, value in required.items() if auth.get(key) != value}
    if mismatches:
        raise RuntimeError(f"authorization contract mismatch: {mismatches}")
    for rel, expected in auth["frozen_file_sha256"].items():
        path = ROOT / rel
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"authorized file changed: {rel}")
    for panel in PANELS:
        release = PUBLIC / "pretruth_release" / panel
        f2 = F2_ROOT / panel / "F2_pretruth"
        if sha256(release / "arrays/PRETRUTH_PREDICTIONS.npz") != auth["panels"][panel]["prediction_sha256"]:
            raise RuntimeError(f"{panel} pretruth prediction changed")
        if sha256(f2 / "MANIFEST.sha256") != auth["panels"][panel]["f2_manifest_sha256"]:
            raise RuntimeError(f"{panel} F2 manifest changed")
    return auth


def write_manifest(directory: Path) -> str:
    lines = []
    for path in sorted(directory.iterdir(), key=lambda p: p.name):
        if path.is_file() and path.name != "MANIFEST.sha256":
            lines.append(f"{sha256(path)}  {path.name}\n")
    (directory / "MANIFEST.sha256").write_text("".join(lines))
    return sha256(directory / "MANIFEST.sha256")


def build_panel(wrapper, panel: str, source: Path, source_hash: str, staging_root: Path,
                authorization_commit: str, auth_hash: str) -> dict:
    helper = wrapper.configure(wrapper.import_helper(), panel)
    # The publication worktree is a later snapshot and does not retain E168's
    # commit as a graph ancestor.  Verify the exact frozen bytes recorded by
    # E170 instead of weakening the check or fabricating ancestry.
    helper.ROOT = Path("/home/yyf/proj")
    run_status = json.loads((PUBLIC / "RUN_STATUS.json").read_text())
    frozen_relatives = [
        "SOURCE_LOCK.json", "MODEL_INPUT_LOCK.json", "STATISTICAL_ANALYSIS_LOCK.json",
        "PREREG_ANALYSIS_PLAN.md", "manifests/E170_DONOR_STATE_ROLES.csv",
        f"manifests/{panel}/E170_{panel}_ROW_ACCESS_MANIFEST.csv",
        f"manifests/{panel}/E170_{panel}_SELECTED_TARGETS.csv",
        f"manifests/{panel}/E170_{panel}_TASK_MANIFEST.csv",
    ]
    for relative in frozen_relatives:
        path = PUBLIC / relative
        expected = run_status["artifact_sha256"].get(relative)
        if expected is None or sha256(path) != expected:
            raise RuntimeError(f"{panel} frozen E170 input changed: {relative}")
    rows, targets, tasks, _roles = helper.validate_manifests()
    f2 = F2_ROOT / panel / "F2_pretruth"
    f2_manifest = helper.verify_manifest(f2)
    f2_attestation = json.loads((f2 / "ACCESS_ATTESTATION.json").read_text())
    if f2_attestation.get("status") != "PASS" or f2_attestation.get("source_full_sha256") != source_hash:
        raise RuntimeError(f"{panel} F2 attestation failed")
    panel_table = pd.read_csv(f2 / "GENE_PANEL.csv", keep_default_na=False)
    if len(panel_table) != 512 or panel_table.panel_index.tolist() != list(range(512)):
        raise RuntimeError(f"{panel} gene panel changed")
    panel_columns = panel_table.source_column_index.to_numpy(np.int64)
    with np.load(f2 / "CONTROL_PROFILES.npz", allow_pickle=False) as archive:
        controls = {key: np.asarray(archive[key], np.float64) for key in archive.files}
    test_rows = rows.loc[rows.x_access_phase.eq("POSTGATE_TEST_TRUTH_X")].copy()
    with h5py.File(source, "r") as handle:
        reader = helper.RowMatrixReader(handle)
        effects, guide_effects, access_rows, guide_table = helper.consume_target_rows(
            reader=reader,
            selected_rows=test_rows,
            panel_columns=panel_columns,
            controls=controls,
            expected_guides=helper.expected_guides_by_target(targets),
            valid_task_ids=set(tasks.task_id.astype(str)),
            stage="SAFECONF_V4_CONFIRMATION_TRUTH",
            batch_size=128,
        )
    if len(effects) != 600 or len(guide_effects) != 1200:
        raise RuntimeError(f"{panel} test truth count changed")
    if set(guide_table.x_access_phase) != {"POSTGATE_TEST_TRUTH_X"}:
        raise RuntimeError(f"{panel} non-test X entered the truth asset")
    test_tasks = tasks.loc[tasks.primary_test_task.astype(str).str.lower().eq("true")].copy()
    if len(test_tasks) != 600 or set(test_tasks.task_id.astype(str)) != set(effects):
        raise RuntimeError(f"{panel} task/effect identity mismatch")

    destination = staging_root / panel
    destination.mkdir()
    helper.save_npz(destination / "TEST_TARGET_EFFECTS.npz", effects)
    helper.save_npz(destination / "TEST_GUIDE_EFFECTS.npz", guide_effects)
    guide_table.to_csv(destination / "TEST_GUIDE_EFFECT_INDEX.csv", index=False)
    test_tasks["truth_effect_asset_key"] = test_tasks.task_id.astype(str)
    test_tasks.to_csv(destination / "TEST_TASKS.csv", index=False)
    access = pd.DataFrame(access_rows).sort_values("metadata_row_index", kind="stable")
    phase_counts = helper.assert_exact_access(access, rows, helper.POSTGATE_PHASES)
    access.to_csv(destination / "ROW_ACCESS_AUDIT.csv", index=False)
    attestation = {
        "status": "PASS",
        "stage": "SAFECONF_V4_E170_CONFIRMATION_TRUTH_BUILD",
        "panel": panel,
        "authorization_commit": authorization_commit,
        "authorization_sha256": auth_hash,
        "source_path": str(source),
        "source_bytes": source.stat().st_size,
        "source_full_sha256": source_hash,
        "source_full_sha256_computed_before_test_x_access": True,
        "f2_manifest_sha256": f2_manifest,
        "logical_x_rows_read": int(len(access)),
        "logical_x_rows_read_by_phase": phase_counts,
        "all_returned_x_rows_read_exactly_once": True,
        "train_or_validation_targeting_x_values_read": 0,
        "column_unseen_non_test_x_values_read": 0,
        "n_test_target_effects": len(effects),
        "n_test_guide_effects": len(guide_effects),
        "test_performance_metrics_computed": 0,
        "guide_weighting": "equal_after_per_guide_raw_sum_then_log1p_1e4_normalization",
    }
    (destination / "ACCESS_ATTESTATION.json").write_text(json.dumps(attestation, indent=2) + "\n")
    manifest = write_manifest(destination)
    return {"panel": panel, "n_tasks": len(effects), "manifest_sha256": manifest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization-commit", required=True)
    parser.add_argument("--branch", required=True)
    args = parser.parse_args()
    auth = verify_authorization(args.authorization_commit, args.branch)
    wrapper = import_wrapper()
    source = Path(auth["source_path"])
    if source.stat().st_size != int(auth["source_bytes"]):
        raise RuntimeError("E170 source byte size changed")
    # The 42 GB source is hashed exactly once before any test targeting X read.
    source_hash = sha256(source)
    if source_hash != auth["source_sha256"]:
        raise RuntimeError("E170 source full SHA-256 changed")

    staging = OUTPUT.with_name(OUTPUT.name + f".staging.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"stale truth staging directory: {staging}")
    staging.mkdir(parents=True)
    try:
        panel_status = []
        auth_hash = sha256(AUTH)
        for panel in PANELS:
            panel_status.append(build_panel(
                wrapper, panel, source, source_hash, staging,
                args.authorization_commit, auth_hash,
            ))
            print(f"[E170-v4 truth] {panel} complete", flush=True)
        root_status = {
            "status": "COMPLETE",
            "all_four_panels_opened_together": True,
            "authorization_commit": args.authorization_commit,
            "authorization_sha256": auth_hash,
            "source_full_sha256": source_hash,
            "n_test_tasks": sum(row["n_tasks"] for row in panel_status),
            "test_performance_metrics_computed": 0,
            "panels": panel_status,
        }
        (staging / "BUILD_STATUS.json").write_text(json.dumps(root_status, indent=2) + "\n")
        write_manifest(staging)
        os.replace(staging, OUTPUT)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(root_status, indent=2), flush=True)


if __name__ == "__main__":
    main()
