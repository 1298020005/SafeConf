#!/usr/bin/env python3
"""Build one sealed E172 panel from the frozen fresh-target manifests.

The audited E168 implementation performs every sparse-row read and asset
check.  This wrapper supplies only E172 identifiers, paths, and immutable
metadata inputs; it does not relax any expression-access boundary.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = ROOT / "tools/scripts/build_e168_primary_cd4_isolated_assets.py"
EXPERIMENT_REL = Path("docs/实验结果/E172_primary_cd4_fresh_targets_20260718")
EXPERIMENT = ROOT / EXPERIMENT_REL
DATA_ROOT = Path("/home/yyf/data/safeconf_external/primary_cd4_perturbseq_2025")
FROZEN_METADATA_COMMIT = "80c1919ac8049cd11ecc36688757a23d89bd9b29"
PANELS = ("Q01", "Q02", "Q03", "Q04")
WRAPPER_REL = Path("tools/scripts/build_e172_primary_cd4_panel_assets.py")
FREEZE_REL = Path("tools/scripts/freeze_e172_primary_cd4_fresh_targets.py")


def import_helper() -> Any:
    spec = importlib.util.spec_from_file_location("safeconf_e172_asset_helper", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import asset helper: {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def configure(helper: Any, panel: str) -> Any:
    if panel not in PANELS:
        raise ValueError(f"unknown E172 panel: {panel}")
    panel_manifest = EXPERIMENT_REL / "manifests" / panel
    helper.EXPERIMENT_ID = f"E172_primary_cd4_fresh_targets::{panel}"
    helper.TASK_PREFIX = f"E172::{panel}"
    helper.PRETRUTH_ASSET_STAGE = f"E172_{panel}_F2_PRETRUTH_ISOLATED_ASSET_BUILD"
    helper.POSTGATE_ASSET_STAGE = f"E172_{panel}_F3_POSTGATE_ISOLATED_TRUTH_BUILD"
    helper.PRETRUTH_GATE_STAGE = f"E172_{panel}_F2_PRETRUTH_GATE"
    helper.EXPERIMENT_REL = EXPERIMENT_REL
    helper.EXPERIMENT = EXPERIMENT
    helper.FROZEN_METADATA_COMMIT = FROZEN_METADATA_COMMIT
    helper.SOURCE_LOCK_REL = EXPERIMENT_REL / "SOURCE_LOCK.json"
    helper.RUN_STATUS_REL = EXPERIMENT_REL / "RUN_STATUS.json"
    helper.MODEL_LOCK_REL = EXPERIMENT_REL / "MODEL_INPUT_LOCK.json"
    helper.STAT_LOCK_REL = EXPERIMENT_REL / "STATISTICAL_ANALYSIS_LOCK.json"
    helper.PLAN_REL = EXPERIMENT_REL / "PREREG_ANALYSIS_PLAN.md"
    helper.DONOR_ROLES_REL = EXPERIMENT_REL / "manifests/E172_DONOR_STATE_ROLES.csv"
    helper.ROW_ACCESS_REL = panel_manifest / f"E172_{panel}_ROW_ACCESS_MANIFEST.csv"
    helper.TARGETS_REL = panel_manifest / f"E172_{panel}_SELECTED_TARGETS.csv"
    helper.TASKS_REL = panel_manifest / f"E172_{panel}_TASK_MANIFEST.csv"
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
    helper.ISOLATED_ROOT = DATA_ROOT / "isolated/E172" / panel
    helper.F2_DIR = helper.ISOLATED_ROOT / "F2_pretruth"
    helper.F3_DIR = helper.ISOLATED_ROOT / "F3_postgate"
    helper.BUILDER_REL = WRAPPER_REL

    original_verify_state = helper.verify_frozen_state
    original_validate_manifests = helper.validate_manifests

    def verify_e172_state() -> Any:
        frozen = original_verify_state()
        source_lock = {
            **frozen.source_lock,
            "local_target_path": frozen.source_lock["source_path"],
            "content_length_bytes": frozen.source_lock["source_bytes"],
            "checksum_crc64nvme_base64": frozen.source_lock[
                "official_crc64nvme_base64"
            ],
        }
        return replace(frozen, source_lock=source_lock)

    def validate_e172_manifests() -> tuple[Any, Any, Any, Any]:
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
        return rows, targets, tasks, roles

    helper.verify_frozen_state = verify_e172_state
    helper.validate_manifests = validate_e172_manifests
    return helper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=PANELS, required=True)
    parser.add_argument("--stage", choices=("pretruth", "postgate"), required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gate-snapshot", type=Path)
    parser.add_argument("--gate-commit")
    parser.add_argument("--branch")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    helper = configure(import_helper(), args.panel)
    if args.self_test:
        helper.self_test()
        print(json.dumps({"status": "PASS", "test": "E172_asset_wrapper+E168_primitives"}))
        return
    if args.stage == "pretruth":
        output = helper.build_pretruth(args.batch_size)
    else:
        if not args.gate_snapshot or not args.gate_commit or not args.branch:
            parser.error("postgate requires --gate-snapshot, --gate-commit, and --branch")
        output = helper.build_postgate(
            batch_size=args.batch_size,
            snapshot_path=args.gate_snapshot,
            gate_commit=args.gate_commit,
            branch=args.branch,
        )
    print(
        json.dumps(
            {"status": "PASS", "panel": args.panel, "stage": args.stage, "output": str(output)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
