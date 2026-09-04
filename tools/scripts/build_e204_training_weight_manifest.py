#!/usr/bin/env python3
"""Build complete source-only E204 weights for all training conditions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TARGETS = ("K562", "RPE1", "hepg2", "jurkat")
EXPECTED_TRAIN_CONDITIONS = 1_366
FROZEN_FORMULA_COMMIT = "5bb3550"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--lambda", dest="weight_lambda", type=float, default=0.5)
    p.add_argument("--clip-low", type=float, default=0.5)
    p.add_argument("--clip-high", type=float, default=2.0)
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def zscore(values: pd.Series) -> pd.Series:
    x = values.astype(float)
    scale = float(x.std(ddof=0))
    if not math.isfinite(scale) or scale == 0:
        return pd.Series(np.zeros(len(x)), index=x.index, dtype=float)
    return (x - float(x.mean())) / scale


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {out}")
    if not 0 < args.clip_low <= 1 <= args.clip_high:
        raise SystemExit("weight clip must contain unit weight")

    # Reuse the already audited physical-view reader.  It verifies each blind
    # H5AD and manifest hash and refuses target perturbation rows.
    sys.path.insert(0, str((ROOT / "tools/scripts").resolve()))
    from build_e201_pretruth_task_base import (  # noqa: PLC0415
        TRAINING_MANIFEST_SHA256,
        TRAINING_VIEW_SHA256,
        source_evidence,
    )

    cache_root = data_root / "txpert_official_20260802/cache"
    blocks: list[pd.DataFrame] = []
    support_blocks: list[pd.DataFrame] = []
    access_blocks: list[pd.DataFrame] = []
    inputs = []

    for target in TARGETS:
        cache = cache_root / f"E201_blind_{target}"
        split_path = cache / "splits/train_test_split.pkl"
        if not split_path.is_file():
            raise SystemExit(f"missing frozen split: {split_path}")
        split = joblib.load(split_path)
        train_conditions = sorted(set(map(str, split.get("train", []))))
        if len(train_conditions) != EXPECTED_TRAIN_CONDITIONS:
            raise SystemExit(
                f"{target}: expected {EXPECTED_TRAIN_CONDITIONS} train conditions, "
                f"found {len(train_conditions)}"
            )
        deltas, support, access = source_evidence(
            target, cache, set(train_conditions)
        )
        if int(access.target_perturbed_expression_rows.sum()) != 0:
            raise SystemExit(f"{target}: target perturbation expression was opened")

        rows = []
        observed_dispersion = []
        for condition in train_conditions:
            context_map = deltas[condition]
            members = np.stack(list(context_map.values()), axis=0)
            center = members.mean(axis=0)
            dispersion = (
                float(np.sqrt(np.mean(np.square(members - center[None, :]))))
                if len(members) >= 2
                else math.nan
            )
            if math.isfinite(dispersion):
                observed_dispersion.append(dispersion)
            condition_support = support.loc[support.condition.eq(condition)]
            n_cells = int(condition_support.n_source_perturbed_cells.sum())
            rows.append(
                {
                    "task_id": f"{target}::{condition}",
                    "target": target,
                    "condition": condition,
                    "n_source_cells": n_cells,
                    "n_source_contexts": len(context_map),
                    "source_delta_dispersion_observed": dispersion,
                    "dispersion_imputed": len(context_map) < 2,
                    "support_context_deficit_source_only": 3 - len(context_map),
                    "log_support_difficulty": -math.log1p(n_cells),
                }
            )
        if not observed_dispersion:
            raise SystemExit(f"{target}: no multi-context source dispersion")
        imputation = float(np.median(observed_dispersion))
        frame = pd.DataFrame(rows)
        frame["source_delta_dispersion"] = frame[
            "source_delta_dispersion_observed"
        ].fillna(imputation)
        frame["z_log_support_difficulty"] = zscore(
            frame["log_support_difficulty"]
        )
        frame["z_context_deficit"] = zscore(
            frame["support_context_deficit_source_only"]
        )
        frame["z_source_dispersion"] = zscore(frame["source_delta_dispersion"])
        frame["difficulty_source_only"] = frame[
            [
                "z_log_support_difficulty",
                "z_context_deficit",
                "z_source_dispersion",
            ]
        ].mean(axis=1)
        frame["task_weight"] = (
            1.0 + args.weight_lambda * frame["difficulty_source_only"]
        ).clip(args.clip_low, args.clip_high)
        frame["dispersion_only_difficulty"] = frame["z_source_dispersion"]
        frame["dispersion_only_weight"] = (
            1.0 + args.weight_lambda * frame["dispersion_only_difficulty"]
        ).clip(args.clip_low, args.clip_high)
        frame["dispersion_imputation_value"] = imputation
        blocks.append(frame)
        support_blocks.append(support)
        access_blocks.append(access)
        inputs.append(
            {
                "target": target,
                "blind_h5ad_sha256": TRAINING_VIEW_SHA256[target],
                "blind_manifest_sha256": TRAINING_MANIFEST_SHA256[target],
                "split_sha256": sha256(split_path),
                "n_train_conditions": len(train_conditions),
                "dispersion_imputation_value": imputation,
            }
        )

    manifest = pd.concat(blocks, ignore_index=True).sort_values(
        ["target", "condition"]
    )
    support_audit = pd.concat(support_blocks, ignore_index=True)
    access_audit = pd.concat(access_blocks, ignore_index=True)
    if (
        len(manifest) != len(TARGETS) * EXPECTED_TRAIN_CONDITIONS
        or manifest.task_id.nunique() != len(manifest)
        or not np.isfinite(
            manifest[
                ["task_weight", "dispersion_only_weight", "difficulty_source_only"]
            ].to_numpy(float)
        ).all()
        or int(access_audit.target_perturbed_expression_rows.sum()) != 0
    ):
        raise SystemExit("combined E204 training-manifest contract failed")

    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "E204_SOURCE_ONLY_TRAINING_WEIGHTS.csv"
    support_path = out / "E204_SOURCE_SUPPORT_AUDIT.csv"
    access_path = out / "E204_SOURCE_ACCESS_AUDIT.csv"
    manifest.to_csv(csv_path, index=False)
    support_audit.to_csv(support_path, index=False)
    access_audit.to_csv(access_path, index=False)
    status = {
        "experiment": "E204_risk_guided_training",
        "stage": "COMPLETE_SOURCE_TRAINING_WEIGHT_MANIFEST",
        "status": "PASS",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "formula_frozen_before_e201_truth_at_commit": FROZEN_FORMULA_COMMIT,
        "implementation_amendment_after_e201_truth": True,
        "target_expression_opened_by_this_builder": 0,
        "target_truth_files_opened_by_this_builder": 0,
        "n_rows": len(manifest),
        "n_targets": len(TARGETS),
        "n_training_conditions_per_target": EXPECTED_TRAIN_CONDITIONS,
        "weight_lambda": args.weight_lambda,
        "weight_clip": [args.clip_low, args.clip_high],
        "inputs": inputs,
        "outputs": [
            {"path": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in (csv_path, support_path, access_path)
        ],
    }
    (out / "E204_TRAINING_WEIGHT_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
