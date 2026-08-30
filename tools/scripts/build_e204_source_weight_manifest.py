#!/usr/bin/env python3
"""Create E204 task weights from source-only E201 metadata.

The script deliberately refuses target-derived columns.  It writes one
condition-level weight per target; the training adapter later broadcasts that
weight to the cells belonging to the condition.  No expression matrix is
opened by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


KEY_COLUMNS = ("task_id", "target", "condition")
SOURCE_COLUMNS = (
    "n_source_cells",
    "n_source_contexts",
    "source_delta_dispersion",
    "dispersion_imputed",
)
FORBIDDEN_COLUMNS = {
    "n_target_cells",
    "n_target_batches",
    "source_mean_delta_row",  # retained in the source table for other audits
    "negative_log_source_cells",  # recompute from the allowed count
    "support_context_deficit",  # recompute with fixed source-context count
    "analysis_stratum",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def zscore(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    std = float(values.std(ddof=0))
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
    return (values - float(values.mean())) / std


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--lambda", dest="weight_lambda", type=float, default=0.5)
    p.add_argument("--clip-low", type=float, default=0.5)
    p.add_argument("--clip-high", type=float, default=2.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    source_path = args.input.resolve()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {out}")
    if not source_path.is_file():
        raise SystemExit(f"missing input: {source_path}")

    frame = pd.read_csv(source_path)
    required = set(KEY_COLUMNS + SOURCE_COLUMNS)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"missing required columns: {missing}")
    # Fail loudly if a future table silently changes the source contract.
    unexpected_forbidden = sorted(FORBIDDEN_COLUMNS & set(frame.columns))
    if not unexpected_forbidden:
        raise SystemExit("input table is missing the expected audit columns")
    if frame["task_id"].duplicated().any():
        raise SystemExit("task_id is not unique")
    if frame["target"].isna().any() or frame["condition"].isna().any():
        raise SystemExit("target/condition contains missing values")

    source = frame.loc[:, list(KEY_COLUMNS) + list(SOURCE_COLUMNS)].copy()
    source["n_source_cells"] = pd.to_numeric(source["n_source_cells"], errors="raise")
    source["n_source_contexts"] = pd.to_numeric(
        source["n_source_contexts"], errors="raise"
    )
    if (source["n_source_cells"] < 0).any() or (source["n_source_contexts"] < 0).any():
        raise SystemExit("source counts must be non-negative")

    dispersion = pd.to_numeric(source["source_delta_dispersion"], errors="coerce")
    impute_value = float(dispersion.median())
    if not np.isfinite(impute_value):
        impute_value = 0.0
    source["dispersion_imputed"] = dispersion.isna()
    source["source_delta_dispersion"] = dispersion.fillna(impute_value)

    # E201 has three source cell backgrounds for every target.  Recompute the
    # deficit rather than trusting a precomputed target-facing column.
    source["support_context_deficit_source_only"] = (
        3.0 - source["n_source_contexts"]
    ).clip(lower=0.0)
    source["log_support_difficulty"] = -np.log1p(source["n_source_cells"])
    source["z_log_support_difficulty"] = zscore(source["log_support_difficulty"])
    source["z_context_deficit"] = zscore(source["support_context_deficit_source_only"])
    source["z_source_dispersion"] = zscore(source["source_delta_dispersion"])
    source["difficulty_source_only"] = source[
        ["z_log_support_difficulty", "z_context_deficit", "z_source_dispersion"]
    ].mean(axis=1)
    source["weight_lambda"] = float(args.weight_lambda)
    source["task_weight"] = (
        1.0 + float(args.weight_lambda) * source["difficulty_source_only"]
    ).clip(lower=args.clip_low, upper=args.clip_high)
    source["dispersion_only_difficulty"] = source["z_source_dispersion"]
    source["dispersion_only_weight"] = (
        1.0 + float(args.weight_lambda) * source["dispersion_only_difficulty"]
    ).clip(lower=args.clip_low, upper=args.clip_high)

    # Keep only stable, auditable columns in the training manifest.
    columns = [
        *KEY_COLUMNS,
        *SOURCE_COLUMNS,
        "support_context_deficit_source_only",
        "log_support_difficulty",
        "z_log_support_difficulty",
        "z_context_deficit",
        "z_source_dispersion",
        "difficulty_source_only",
        "task_weight",
        "dispersion_only_difficulty",
        "dispersion_only_weight",
    ]
    manifest = source.loc[:, columns].sort_values(["target", "condition"]).reset_index(drop=True)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "E204_SOURCE_ONLY_TASK_WEIGHTS.csv"
    manifest.to_csv(csv_path, index=False)

    metadata = {
        "experiment": "E204_risk_guided_training",
        "status": "SOURCE_ONLY_COMPLETE",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_path": str(source_path),
        "input_sha256": sha256(source_path),
        "rows": int(len(manifest)),
        "targets": sorted(manifest["target"].unique().tolist()),
        "allowed_input_columns": list(KEY_COLUMNS + SOURCE_COLUMNS),
        "forbidden_target_derived_columns": sorted(FORBIDDEN_COLUMNS),
        "target_expression_opened": False,
        "lambda": float(args.weight_lambda),
        "clip": [float(args.clip_low), float(args.clip_high)],
        "dispersion_imputation_value": impute_value,
        "formula": "clip(1 + lambda * mean(z(-log1p(n_source_cells)), z(3-n_source_contexts), z(source_delta_dispersion)), low, high)",
        "csv_sha256": sha256(csv_path),
    }
    (out / "E204_SOURCE_ONLY_TASK_WEIGHTS.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
