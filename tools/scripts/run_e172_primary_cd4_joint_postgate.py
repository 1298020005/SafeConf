#!/usr/bin/env python3
"""Run the frozen joint E172 evaluation only after all four PASS gates."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = ROOT / "tools/scripts/run_e170_primary_cd4_joint_postgate.py"
EXPERIMENT = ROOT / "docs/实验结果/E172_primary_cd4_fresh_targets_20260718"
DATA_ROOT = Path("/home/yyf/data/safeconf_external/primary_cd4_perturbseq_2025")
PANELS = ("Q01", "Q02", "Q03", "Q04")


def import_helper() -> Any:
    spec = importlib.util.spec_from_file_location("safeconf_e172_joint_helper", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import joint evaluator helper: {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def configure(helper: Any) -> Any:
    helper.SCRIPT = Path(__file__).resolve()
    helper.EXPERIMENT_CODE = "E172"
    helper.EXPERIMENT_ID = "E172_primary_cd4_fresh_targets"
    helper.ISOLATED_NAMESPACE = "E172"
    helper.TASK_PREFIX = "E172"
    helper.PRETRUTH_SCHEMA_PREFIX = "safeconf_e172"
    helper.PRETRUTH_REPORT_PREFIX = "E172"
    helper.EXPECTED_G4_RISK_ESTIMATOR = "leave_one_seed_out_family_mean"
    helper.STATUS_SCHEMA = "safeconf_e172_joint_postgate_result_v1"
    helper.JOINT_STAGE = "E172_JOINT_F3_POSTGATE_FORMAL_EVALUATION"
    helper.JOINT_REPORT_NAME = "E172_JOINT_POSTGATE_REPORT.md"
    helper.JOINT_REPORT_TITLE = "E172｜修正 seed gate 后的未读目标确认"
    helper.EXPERIMENT = EXPERIMENT
    helper.DATA_ROOT = DATA_ROOT
    helper.RELEASE = EXPERIMENT / "postgate_release"
    helper.STAGING = EXPERIMENT / ".postgate_release.staging"
    helper.PANELS = PANELS
    helper.BOOTSTRAP_SEED = 2026071821
    helper.PERMUTATION_SEED = 2026071822
    helper.PRETRUTH_WRAPPER = ROOT / "tools/scripts/run_e172_primary_cd4_panel_pretruth.py"
    helper.PRETRUTH_HELPER = ROOT / "tools/scripts/run_e168_primary_cd4_pretruth.py"
    helper.ASSET_WRAPPER = ROOT / "tools/scripts/build_e172_primary_cd4_panel_assets.py"
    helper.ASSET_HELPER = ROOT / "tools/scripts/build_e168_primary_cd4_isolated_assets.py"
    helper.POSTGATE_HELPER = ROOT / "tools/scripts/run_e168_primary_cd4_postgate.py"
    helper.FREEZE_SCRIPT = ROOT / "tools/scripts/freeze_e172_primary_cd4_fresh_targets.py"
    helper.STAT_LOCK = EXPERIMENT / "STATISTICAL_ANALYSIS_LOCK.json"
    helper.MODEL_LOCK = EXPERIMENT / "MODEL_INPUT_LOCK.json"
    helper.ALL_TARGETS = EXPERIMENT / "manifests/E172_ALL_SELECTED_TARGETS.csv"
    helper.ALL_TASKS = EXPERIMENT / "manifests/E172_ALL_TASKS.csv"
    helper.EXECUTION_PLAN = EXPERIMENT / "PRETRUTH_CODE_FREEZE_PLAN.md"
    helper.CORRECTION_LOG = EXPERIMENT / "PRETRUTH_RUNTIME_CORRECTION_LOG.md"
    helper.PRIOR_TARGET_MANIFESTS = (
        ROOT
        / "docs/实验结果/E168_primary_human_cd4_fresh_confirmation_20260716"
        / "manifests/E168_SELECTED_TARGETS.csv",
        ROOT
        / "docs/实验结果/E170_primary_cd4_multipanel_precision_20260718"
        / "manifests/E170_ALL_SELECTED_TARGETS.csv",
    )
    return helper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-commit")
    parser.add_argument("--branch")
    parser.add_argument("--synthetic-test-only", action="store_true")
    args = parser.parse_args()
    helper = configure(import_helper())
    if args.synthetic_test_only:
        tests = helper.synthetic_tests()
        print(tests.to_string(index=False))
        if len(tests) != 7 or not tests.passed.astype(bool).all():
            raise SystemExit(2)
        return
    if not args.gate_commit or not args.branch:
        parser.error("formal evaluation requires --gate-commit and --branch")
    result = helper.run_formal(args.gate_commit, args.branch)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
