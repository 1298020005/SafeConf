#!/usr/bin/env python3
"""Build physically isolated E182 F2 assets from the frozen metadata commit."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
BASE_BUILDER = ROOT / "tools/scripts/build_e180_xucao_pretruth_assets.py"
EXPERIMENT_REL = Path(
    "docs/实验结果/E182_gse225807_registered_family_20260724"
)
DATA_ROOT = Path("/home/yyf/data/safeconf_e182_gse225807")
F2_ROOT = DATA_ROOT / "isolated/F2_pretruth"


class IntegrityError(RuntimeError):
    """The E182 physical access boundary or frozen input changed."""


def import_script(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise IntegrityError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def configure(base: Any, frozen_commit: str) -> None:
    base.EXPERIMENT_REL = EXPERIMENT_REL
    base.EXPERIMENT = ROOT / EXPERIMENT_REL
    base.FROZEN_METADATA_COMMIT = frozen_commit
    base.SOURCE_LOCK_REL = EXPERIMENT_REL / "SOURCE_LOCK.json"
    base.RUN_STATUS_REL = EXPERIMENT_REL / "RUN_STATUS.json"
    base.MODEL_LOCK_REL = EXPERIMENT_REL / "MODEL_INPUT_LOCK.json"
    base.STAT_LOCK_REL = EXPERIMENT_REL / "STATISTICAL_ANALYSIS_LOCK.json"
    base.PLAN_REL = EXPERIMENT_REL / "PREREG_ANALYSIS_PLAN.md"
    base.TARGETS_REL = EXPERIMENT_REL / "manifests/E182_SELECTED_TARGETS.csv"
    base.TASKS_REL = EXPERIMENT_REL / "manifests/E182_GUIDE_TASK_MANIFEST.csv"
    base.BUILDER_REL = RUNNER.relative_to(ROOT)
    base.FROZEN_INPUTS = (
        base.SOURCE_LOCK_REL,
        base.RUN_STATUS_REL,
        base.MODEL_LOCK_REL,
        base.STAT_LOCK_REL,
        base.PLAN_REL,
        base.TARGETS_REL,
        base.TASKS_REL,
    )
    base.DATA_ROOT = DATA_ROOT
    base.ISOLATED_ROOT = DATA_ROOT / "isolated"
    base.F2_DIR = F2_ROOT


def relabel_attestation(base: Any, frozen_commit: str) -> str:
    path = F2_ROOT / "ACCESS_ATTESTATION.json"
    value = json.loads(path.read_text())
    value.update(
        {
            "schema": "safeconf_e182_f2_asset_attestation_v1",
            "experiment": "E182_gse225807_registered_family",
            "generated_from_metadata_commit": frozen_commit,
            "base_builder": str(BASE_BUILDER.relative_to(ROOT)),
            "base_builder_sha256": base.sha256_file(BASE_BUILDER),
            "guide_replicates_are_primary_task_units": True,
            "negative_guide_controls_are_pooled": True,
            "learned_or_adaptive_upper_model_fitted": False,
        }
    )
    base.atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return base.write_manifest(F2_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-metadata-commit", required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    base = import_script("e182_base_asset_builder", BASE_BUILDER)
    configure(base, args.frozen_metadata_commit)
    if args.validate_only:
        result = base.validate_existing()
        attestation = json.loads(
            (F2_ROOT / "ACCESS_ATTESTATION.json").read_text()
        )
        if attestation.get("experiment") != "E182_gse225807_registered_family":
            raise IntegrityError("E182 F2 attestation label changed")
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return

    result = base.build(args.batch_size)
    manifest_sha = relabel_attestation(base, args.frozen_metadata_commit)
    result["experiment"] = "E182_gse225807_registered_family"
    result["manifest_sha256"] = manifest_sha
    result["builder_wrapper"] = str(RUNNER.relative_to(ROOT))
    result["base_builder"] = str(BASE_BUILDER.relative_to(ROOT))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
