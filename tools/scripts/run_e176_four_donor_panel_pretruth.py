#!/usr/bin/env python3
"""Train five-seed scGPT/GEARS families for one E176 panel without test truth."""

from __future__ import annotations

import argparse
from itertools import combinations
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = ROOT / "tools/scripts/run_e168_primary_cd4_pretruth.py"
ASSET_HELPER = ROOT / "tools/scripts/build_e168_primary_cd4_isolated_assets.py"
ASSET_WRAPPER = ROOT / "tools/scripts/build_e176_four_donor_panel_assets.py"
FREEZE_SCRIPT = ROOT / "tools/scripts/freeze_e176_four_donor_fresh_confirmation.py"
EXPERIMENT = ROOT / "docs/实验结果/E176_four_donor_fresh_confirmation_20260719"
E174_METHOD = ROOT / "docs/实验结果/E174_rotated_donor_conformal_certificate_20260719/method_development/METHOD_GATE_SNAPSHOT.json"
E175_GATE = ROOT / "docs/实验结果/E175_e174_seed_extension_development_20260719/aggregate/RUN_STATUS.json"
DATA_ROOT = Path("/home/yyf/data/safeconf_external/primary_cd4_perturbseq_2025")
PANELS = ("H01", "H02", "H03", "H04")
SEEDS = (3407, 3408, 3409, 3410, 3411)


def import_helper() -> Any:
    spec = importlib.util.spec_from_file_location("safeconf_e176_pretruth_helper", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import pretruth helper: {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def five_seed_stability(helper: Any, risks: np.ndarray, seed: int, n_boot: int) -> dict[str, Any]:
    values = np.asarray(risks, float)
    if values.ndim != 2 or values.shape[0] < 3:
        raise helper.IntegrityFailure("G4 requires at least three aligned risk vectors")
    pairs = list(combinations(range(values.shape[0]), 2))
    correlations = np.asarray(
        [helper.spearman(values[a], values[b]) for a, b in pairs], dtype=float
    )
    median = float(np.nanmedian(correlations))
    rng = np.random.default_rng(seed)
    boot: list[float] = []
    for _ in range(n_boot):
        take = rng.integers(0, values.shape[1], values.shape[1])
        draw = [helper.spearman(values[a, take], values[b, take]) for a, b in pairs]
        if np.isfinite(draw).all():
            boot.append(float(np.median(draw)))
    lower = float(np.quantile(boot, 0.025)) if boot else float("nan")
    upper = float(np.quantile(boot, 0.975)) if boot else float("nan")
    return {
        "n_leave_one_seed_out_estimators": int(values.shape[0]),
        "n_pairwise_correlations": len(pairs),
        "minimum_pairwise_spearman": float(np.nanmin(correlations)),
        "median_pairwise_spearman": median,
        "maximum_pairwise_spearman": float(np.nanmax(correlations)),
        "kendall_w": helper.kendall_w(values),
        "bootstrap_valid": len(boot),
        "bootstrap_ci95_lower": lower,
        "bootstrap_ci95_upper": upper,
        "passed": bool(
            math.isfinite(median) and median >= 0.5
            and math.isfinite(lower) and lower > 0
        ),
    }


def configure(helper: Any, panel: str) -> Any:
    if panel not in PANELS:
        raise ValueError(f"unknown E176 panel: {panel}")
    panel_manifest = EXPERIMENT / "manifests" / panel
    helper.RUNNER = Path(__file__).resolve()
    helper.EXPERIMENT_ID = f"E176_four_donor_fresh_confirmation::{panel}"
    helper.SNAPSHOT_SCHEMA = f"safeconf_e176_{panel.lower()}_pretruth_gate_snapshot_v1"
    helper.PRETRUTH_GATE_STAGE = f"E176_{panel}_F2_PRETRUTH_GATE"
    helper.PRETRUTH_ASSET_STAGE = f"E176_{panel}_F2_PRETRUTH_ISOLATED_ASSET_BUILD"
    helper.EXPECTED_ASSET_DIR_NAME = "F2_pretruth"
    helper.PRETRUTH_REPORT_NAME = f"E176_{panel}_PRETRUTH_REPORT.md"
    helper.PRETRUTH_REPORT_TITLE = f"E176 {panel} five-seed pretruth gate"
    helper.G4_SEED_NAMESPACE = f"E176_{panel}_FIVE_SEED_LOO_G4"
    helper.G4_RISK_MODE = "leave_one_seed_out_family_mean"
    helper.SEEDS = SEEDS
    helper.OUT = EXPERIMENT
    helper.DEFAULT_ASSETS = DATA_ROOT / "isolated/E176" / panel / "F2_pretruth"
    helper.RELEASE = EXPERIMENT / "pretruth_release" / panel
    helper.STAGING = EXPERIMENT / f".pretruth_release.{panel}.staging"
    helper.TASK_MANIFEST = panel_manifest / f"E176_{panel}_TASK_MANIFEST.csv"
    helper.SELECTED_TARGETS = panel_manifest / f"E176_{panel}_SELECTED_TARGETS.csv"
    helper.DONOR_ROLES = panel_manifest / f"E176_{panel}_DONOR_STATE_ROLES.csv"
    helper.MODEL_LOCK = EXPERIMENT / "MODEL_INPUT_LOCK.json"
    helper.SOURCE_LOCK = EXPERIMENT / "SOURCE_LOCK.json"
    helper.ANALYSIS_PLAN = EXPERIMENT / "PREREG_ANALYSIS_PLAN.md"
    helper.ASSET_BUILDER = ASSET_WRAPPER

    def g4(values: np.ndarray, seed: int, n_boot: int = helper.G4_BOOTSTRAPS) -> dict[str, Any]:
        return five_seed_stability(helper, values, seed, n_boot)

    helper.g4_stability = g4

    def formal_input_audit(
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
            helper.TASK_MANIFEST,
            helper.SELECTED_TARGETS,
            helper.DONOR_ROLES,
            helper.MODEL_LOCK,
            helper.SOURCE_LOCK,
            helper.ANALYSIS_PLAN,
            E174_METHOD,
            E175_GATE,
            helper.PROTOCOL,
            helper.E65_SCRIPT,
        ]
        hashes = [helper.require_committed(path, head) for path in files]
        lock = json.loads(helper.MODEL_LOCK.read_text())
        seed_gate = json.loads(E175_GATE.read_text())
        method = json.loads(E174_METHOD.read_text())
        if lock.get("model_seeds") != list(SEEDS):
            raise helper.IntegrityFailure("E176 five-seed lock changed")
        if lock.get("g4_seed_estimator") != \
                "five_leave_one_seed_out_four_seed_family_means":
            raise helper.IntegrityFailure("E176 G4 estimator lock changed")
        if seed_gate.get("decision") != "FIVE_SEED_GATE_READY_FOR_NEW_TARGET_PROTOCOL":
            raise helper.IntegrityFailure("E175 development gate changed")
        if method.get("selected_model_spec") != {
            "ensemble_rmse": "magnitude", "pair_mean_rmse": "magnitude"
        }:
            raise helper.IntegrityFailure("frozen prior-data method changed")
        for path_text, expected in lock["scgpt_checkpoint_files"].items():
            path = Path(path_text)
            if helper.sha256_file(path) != expected:
                raise helper.IntegrityFailure(f"scGPT checkpoint changed: {path}")
            hashes.append({"path": str(path), "bytes": path.stat().st_size, "sha256": expected})
        go_lock = lock["gears_external_go_file"]
        go_path = Path(go_lock["path"])
        if helper.sha256_file(go_path) != go_lock["sha256"]:
            raise helper.IntegrityFailure("GEARS GO prior changed")
        hashes.append({
            "path": str(go_path), "bytes": go_path.stat().st_size,
            "sha256": go_lock["sha256"],
        })
        return head, branch, remote_heads, hashes

    helper.formal_input_audit = formal_input_audit
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
