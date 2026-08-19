#!/usr/bin/env python3
"""Train and seal one E174 panel without loading held-out-donor targeting truth."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = ROOT / "tools/scripts/run_e168_primary_cd4_pretruth.py"
ASSET_HELPER = ROOT / "tools/scripts/build_e168_primary_cd4_isolated_assets.py"
ASSET_WRAPPER = ROOT / "tools/scripts/build_e174_rotated_donor_panel_assets.py"
FREEZE_SCRIPT = ROOT / "tools/scripts/freeze_e174_rotated_donor_conformal_certificate.py"
COMMON = ROOT / "tools/scripts/e174_conformal_common.py"
METHOD_SCRIPT = ROOT / "tools/scripts/run_e174_method_development.py"
CALIBRATION_RUNNER = ROOT / "tools/scripts/run_e174_joint_calibration.py"
FINAL_EVALUATOR = ROOT / "tools/scripts/run_e174_final_evaluation.py"
EXPERIMENT = ROOT / "docs/实验结果/E174_rotated_donor_conformal_certificate_20260719"
METHOD_SNAPSHOT = EXPERIMENT / "method_development/METHOD_GATE_SNAPSHOT.json"
METHOD_STATUS = EXPERIMENT / "method_development/RUN_STATUS.json"
METHOD_MANIFEST = EXPERIMENT / "method_development/MANIFEST.sha256"
EXECUTION_PLAN = EXPERIMENT / "EXECUTION_PLAN.md"
RUNTIME_LOG = EXPERIMENT / "RUNTIME_LOG.md"
DATA_ROOT = Path("/home/yyf/data/safeconf_external/primary_cd4_perturbseq_2025")
PANELS = ("R01", "R02", "R03", "R04")


def import_helper() -> Any:
    spec = importlib.util.spec_from_file_location("safeconf_e174_pretruth_helper", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import pretruth helper: {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def configure(helper: Any, panel: str) -> Any:
    if panel not in PANELS:
        raise ValueError(f"unknown E174 panel: {panel}")
    panel_manifest = EXPERIMENT / "manifests" / panel
    helper.RUNNER = Path(__file__).resolve()
    helper.EXPERIMENT_ID = f"E174_rotated_donor_conformal_certificate::{panel}"
    helper.SNAPSHOT_SCHEMA = f"safeconf_e174_{panel.lower()}_pretruth_gate_snapshot_v1"
    helper.PRETRUTH_GATE_STAGE = f"E174_{panel}_F2_PRETRUTH_GATE"
    helper.PRETRUTH_ASSET_STAGE = f"E174_{panel}_F2_PRETRUTH_ISOLATED_ASSET_BUILD"
    helper.EXPECTED_ASSET_DIR_NAME = "F2_pretruth"
    helper.PRETRUTH_REPORT_NAME = f"E174_{panel}_PRETRUTH_REPORT.md"
    helper.PRETRUTH_REPORT_TITLE = f"E174 {panel} pretruth gate"
    helper.G4_SEED_NAMESPACE = f"E174_{panel}_LOO_G4"
    helper.G4_RISK_MODE = "leave_one_seed_out_family_mean"
    helper.OUT = EXPERIMENT
    helper.DEFAULT_ASSETS = DATA_ROOT / "isolated/E174" / panel / "F2_pretruth"
    helper.RELEASE = EXPERIMENT / "pretruth_release" / panel
    helper.STAGING = EXPERIMENT / f".pretruth_release.{panel}.staging"
    helper.TASK_MANIFEST = panel_manifest / f"E174_{panel}_TASK_MANIFEST.csv"
    helper.SELECTED_TARGETS = panel_manifest / f"E174_{panel}_SELECTED_TARGETS.csv"
    helper.DONOR_ROLES = EXPERIMENT / "manifests/E174_DONOR_STATE_ROLES.csv"
    helper.MODEL_LOCK = EXPERIMENT / "MODEL_INPUT_LOCK.json"
    helper.SOURCE_LOCK = EXPERIMENT / "SOURCE_LOCK.json"
    helper.ANALYSIS_PLAN = EXPERIMENT / "PREREG_ANALYSIS_PLAN.md"
    helper.ASSET_BUILDER = ASSET_WRAPPER

    def formal_e174_input_audit(
        asset_root: Path,
    ) -> tuple[str, str, dict[str, str], list[dict[str, Any]]]:
        head = helper.git_head()
        branch, remote_heads = helper.verify_dual_remote_contains_head(head)
        files = [
            helper.RUNNER,
            ASSET_WRAPPER,
            ASSET_HELPER,
            HELPER_PATH,
            FREEZE_SCRIPT,
            COMMON,
            METHOD_SCRIPT,
            CALIBRATION_RUNNER,
            FINAL_EVALUATOR,
            EXECUTION_PLAN,
            RUNTIME_LOG,
            helper.TASK_MANIFEST,
            helper.SELECTED_TARGETS,
            helper.DONOR_ROLES,
            helper.MODEL_LOCK,
            helper.SOURCE_LOCK,
            helper.ANALYSIS_PLAN,
            METHOD_SNAPSHOT,
            METHOD_STATUS,
            METHOD_MANIFEST,
            helper.PROTOCOL,
            helper.E65_SCRIPT,
        ]
        hashes = [helper.require_committed(path, head) for path in files]
        lock = json.loads(helper.MODEL_LOCK.read_text())
        method = json.loads(METHOD_SNAPSHOT.read_text())
        method_status = json.loads(METHOD_STATUS.read_text())
        if helper.sha256_file(FREEZE_SCRIPT) != lock["metadata_freeze_script_sha256"]:
            raise helper.IntegrityFailure("E174 metadata-freeze script changed after selection")
        if lock.get("g4_seed_estimator") != "three_leave_one_seed_out_two_seed_family_means":
            raise helper.IntegrityFailure("E174 model lock no longer authorizes LOO G4")
        if lock.get("primary_method") != "pair_lower_certificate_plus_split_conformal_upper_bound":
            raise helper.IntegrityFailure("E174 primary method lock changed")
        if method.get("selected_model_spec") != {
            "ensemble_rmse": "magnitude",
            "pair_mean_rmse": "magnitude",
        }:
            raise helper.IntegrityFailure("prior-data fallback selection changed")
        if (
            method.get("e174_expression_x_values_read") != 0
            or method.get("e174_calibration_truth_used") is not False
            or method.get("e174_evaluation_truth_used") is not False
        ):
            raise helper.IntegrityFailure("method selection was not truth blind for E174")
        if helper.sha256_file(METHOD_MANIFEST) != method_status.get("manifest_sha256"):
            raise helper.IntegrityFailure("method development manifest changed")
        for path_text, expected in lock["scgpt_checkpoint_files"].items():
            path = Path(path_text)
            if helper.sha256_file(path) != expected:
                raise helper.IntegrityFailure(f"scGPT checkpoint changed: {path}")
            hashes.append({"path": str(path), "bytes": path.stat().st_size, "sha256": expected})
        go_lock = lock["gears_external_go_file"]
        go_path = Path(go_lock["path"])
        if helper.sha256_file(go_path) != go_lock["sha256"]:
            raise helper.IntegrityFailure("GEARS GO prior changed")
        hashes.append(
            {"path": str(go_path), "bytes": go_path.stat().st_size, "sha256": go_lock["sha256"]}
        )
        return head, branch, remote_heads, hashes

    helper.formal_input_audit = formal_e174_input_audit
    return helper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=PANELS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--synthetic-test-only", action="store_true")
    args = parser.parse_args()
    helper = configure(import_helper(), args.panel)
    if args.synthetic_test_only:
        tests = helper.synthetic_tests()
        print(tests.to_string(index=False))
        if len(tests) != 10 or not tests.passed.astype(bool).all():
            raise SystemExit(2)
        return
    helper.RELEASE.parent.mkdir(parents=True, exist_ok=True)
    result = helper.run_formal(args.asset_root or helper.DEFAULT_ASSETS, args.device)
    print(json.dumps({"panel": args.panel, **result}, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
