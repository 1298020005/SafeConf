#!/usr/bin/env python3
"""Training-only audit of the Wessels endogenous-expression boundary.

This diagnostic is deliberately narrower than E161.  It materializes only
the E160 training rows and the first 20,639 columns, compares fixed prefix
sums with obs[ncounts], and never indexes validation/test/excluded expression.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
CONTRACT = (
    ROOT
    / "docs"
    / "实验结果"
    / "E161a_wessels_feature_boundary_audit_20260715"
    / "ANALYSIS_CONTRACT.md"
)
OUT = CONTRACT.parent
E161_RUNNER = ROOT / "tools" / "scripts" / "run_e161_wessels_trainval_preprocess.py"
RAW = Path(
    "/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/"
    "WesselsSatija2023.h5ad"
)

EXPECTED_RAW_SHA256 = "5da0485aed81b23bda57b4a7b4510a394682d54911416db89b4846ff6dd34732"
EXPECTED_FULL_AXIS_SHA256 = "dea725a87c973ca15590b08b309df3a926dc0233391cb2df76518c847229e780"
EXPECTED_ENDOGENOUS_AXIS_SHA256 = "dbed3dad178ea500b01625abf5121c9ee17bdd501b87d2fcdede0b6bade654e7"
EXPECTED_ENGINEERED_AXIS_SHA256 = "103c2df8585646aa6dccde85866353889a699420b5536157b8babbd9b9aec554"
EXPECTED_GUIDE_AXIS_SHA256 = "9088328f4ac6b2a1b109c254f0068504d25618478383fcbb3f43be8e59dd06d2"
EXPECTED_ALL_EXCLUDED_SHA256 = "e6e54ba5c0f63d62b599754ab3866da7cdf8194be4dfefd46dabc7d6a73e8116"
EXPECTED_TRAIN_CELLS = 11_779
ENDOGENOUS_FEATURES = 20_631
AUDITED_FEATURES = 20_639
FULL_FEATURES = 21_052
ENGINEERED_NAMES = [
    "eGFP",
    "Blast",
    "Cas9",
    "Puro",
    "Cas13d",
    "AsCas12a",
    "MeCP2",
    "KRAB",
]
OUTPUTS = [
    "RUN_STATUS.json",
    "CANDIDATE_BOUNDARY_AUDIT.csv",
    "ENGINEERED_CONSTRUCT_COUNTS.csv",
    "ACCESS_LEDGER.json",
    "REPORT.md",
    "RESULTS_SHA256.csv",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def axis_hash(names: list[str] | np.ndarray) -> str:
    return hashlib.sha256(("\n".join(map(str, names)) + "\n").encode("utf-8")).hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def load_e161() -> Any:
    spec = importlib.util.spec_from_file_location("safeconf_e161_locked", E161_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the locked E161 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_clean_output_slot() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    present = [name for name in OUTPUTS if (OUT / name).exists()]
    if present:
        raise RuntimeError(f"Refusing to overwrite prior E161a outputs: {present}")


def read_train_expression_once(
    frozen: dict[str, Any],
) -> tuple[sp.csr_matrix, dict[str, Any]]:
    rows = np.asarray(frozen["split_rows"]["train"], dtype=np.int64)
    if len(rows) != EXPECTED_TRAIN_CELLS or not np.all(rows[:-1] < rows[1:]):
        raise RuntimeError("Frozen training row index changed")
    intersections = {
        role: int(np.intersect1d(rows, frozen["split_rows"][role]).size)
        for role in ("val", "test", "excluded")
    }
    if any(intersections.values()):
        raise RuntimeError(f"Training rows intersect a sealed role: {intersections}")

    raw = ad.read_h5ad(RAW, backed="r")
    try:
        # Sole expression indexing statement in E161a.
        matrix = raw.X[rows, :AUDITED_FEATURES]
    finally:
        raw.file.close()
    if not sp.issparse(matrix):
        matrix = sp.csr_matrix(matrix)
    matrix = matrix.tocsr()
    if matrix.shape != (EXPECTED_TRAIN_CELLS, AUDITED_FEATURES):
        raise RuntimeError(f"Unexpected audit matrix shape: {matrix.shape}")
    if not np.issubdtype(matrix.dtype, np.integer):
        raise RuntimeError(f"Expected integer raw counts, observed {matrix.dtype}")
    if matrix.nnz and int(matrix.data.min()) < 0:
        raise RuntimeError("Negative count in training expression")
    ledger = {
        "split": "train",
        "rows_indexed": int(matrix.shape[0]),
        "columns_indexed": int(matrix.shape[1]),
        "column_interval_half_open": [0, AUDITED_FEATURES],
        "validation_rows_indexed": 0,
        "test_rows_indexed": 0,
        "excluded_rows_indexed": 0,
        "guide_or_barcode_columns_indexed": 0,
        "sealed_role_intersections": intersections,
        "row_index_sha256": hashlib.sha256(rows.tobytes()).hexdigest(),
        "X_materialized": True,
        "transformation": "none_raw_integer_counts",
    }
    return matrix, ledger


def main() -> None:
    check_clean_output_slot()
    started = datetime.now().astimezone().isoformat()
    e161 = load_e161()
    frozen = e161.metadata_preflight()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    committed_inputs = {
        "runner": e161.git_blob_gate(RUNNER, head),
        "contract": e161.git_blob_gate(CONTRACT, head),
        "e161_runner": e161.git_blob_gate(E161_RUNNER, head),
    }
    raw_identity = e161.hash_raw_once(RAW)
    if raw_identity["sha256"] != EXPECTED_RAW_SHA256:
        raise RuntimeError("Raw Wessels hash changed")

    var_names = np.asarray(frozen["raw_var_names"], dtype=str)
    if len(var_names) != FULL_FEATURES or axis_hash(var_names) != EXPECTED_FULL_AXIS_SHA256:
        raise RuntimeError("Full feature axis changed")
    if axis_hash(var_names[:ENDOGENOUS_FEATURES]) != EXPECTED_ENDOGENOUS_AXIS_SHA256:
        raise RuntimeError("Endogenous candidate axis changed")
    if list(var_names[ENDOGENOUS_FEATURES:AUDITED_FEATURES]) != ENGINEERED_NAMES:
        raise RuntimeError("Engineered-construct names changed")
    if axis_hash(var_names[ENDOGENOUS_FEATURES:AUDITED_FEATURES]) != EXPECTED_ENGINEERED_AXIS_SHA256:
        raise RuntimeError("Engineered-construct axis changed")
    if axis_hash(var_names[AUDITED_FEATURES:]) != EXPECTED_GUIDE_AXIS_SHA256:
        raise RuntimeError("Guide/barcode axis changed")
    if axis_hash(var_names[ENDOGENOUS_FEATURES:]) != EXPECTED_ALL_EXCLUDED_SHA256:
        raise RuntimeError("Combined excluded-candidate axis changed")

    source_text = RUNNER.read_text(encoding="utf-8")
    # Split the sentinel literal so this guard does not count itself.
    if source_text.count("raw." + "X[") != 1:
        raise RuntimeError("E161a must contain exactly one raw expression indexing site")

    matrix, ledger = read_train_expression_once(frozen)
    train_rows = np.asarray(frozen["split_rows"]["train"], dtype=np.int64)
    ncounts = pd.to_numeric(
        frozen["obs"].iloc[train_rows]["ncounts"], errors="raise"
    ).to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(ncounts)):
        raise RuntimeError("Non-finite obs[ncounts] in training rows")

    rows: list[dict[str, Any]] = []
    exact_boundaries: list[int] = []
    for boundary in range(ENDOGENOUS_FEATURES, AUDITED_FEATURES + 1):
        library = np.asarray(matrix[:, :boundary].sum(axis=1)).reshape(-1).astype(np.float64)
        delta = library - ncounts
        mismatch = int(np.count_nonzero(delta))
        if mismatch == 0:
            exact_boundaries.append(boundary)
        rows.append(
            {
                "prefix_features": boundary,
                "last_feature": str(var_names[boundary - 1]),
                "mismatched_train_cells": mismatch,
                "max_abs_delta_vs_obs_ncounts": float(np.max(np.abs(delta))),
                "min_delta": float(np.min(delta)),
                "max_delta": float(np.max(delta)),
                "sum_delta": float(np.sum(delta)),
                "exact_match_all_train_cells": mismatch == 0,
            }
        )
    candidates = pd.DataFrame(rows)

    construct_rows: list[dict[str, Any]] = []
    for offset, name in enumerate(ENGINEERED_NAMES):
        col = np.asarray(matrix[:, ENDOGENOUS_FEATURES + offset].toarray()).reshape(-1)
        construct_rows.append(
            {
                "feature_index_zero_based": ENDOGENOUS_FEATURES + offset,
                "feature": name,
                "total_count_train": int(col.sum()),
                "nonzero_train_cells": int(np.count_nonzero(col)),
                "max_count_per_train_cell": int(col.max()),
                "mean_count_per_train_cell": float(col.mean()),
            }
        )
    constructs = pd.DataFrame(construct_rows)

    status = {
        "experiment": "E161a",
        "status": "completed_diagnostic",
        "started_at": started,
        "completed_at": datetime.now().astimezone().isoformat(),
        "git_head": head,
        "committed_inputs": committed_inputs,
        "raw_identity": raw_identity,
        "candidate_boundaries": list(range(ENDOGENOUS_FEATURES, AUDITED_FEATURES + 1)),
        "exact_matching_boundaries": exact_boundaries,
        "unique_exact_boundary": exact_boundaries[0] if len(exact_boundaries) == 1 else None,
        "train_cells_read": EXPECTED_TRAIN_CELLS,
        "validation_expression_rows_read": 0,
        "test_expression_rows_read": 0,
        "excluded_expression_rows_read": 0,
        "guide_or_barcode_columns_read": 0,
    }
    report = f"""# E161a 结果：Wessels 表达轴边界核查

- 训练细胞：{EXPECTED_TRAIN_CELLS:,}
- validation/test/excluded 表达访问：0 / 0 / 0
- 后 413 个 guide/barcode 表达列访问：0
- 与 `obs[ncounts]` 精确一致的候选边界：{', '.join(map(str, exact_boundaries)) if exact_boundaries else '无'}
- 唯一精确边界：{status['unique_exact_boundary'] if status['unique_exact_boundary'] is not None else '无'}

本结果仅回答表达轴边界，不选择 HVG、不拟合 PCA、不训练预测模型。后续 E161 修订必须保留本次访问台账和全部候选比较。
"""

    atomic_write(OUT / "CANDIDATE_BOUNDARY_AUDIT.csv", candidates.to_csv(index=False).encode("utf-8"))
    atomic_write(OUT / "ENGINEERED_CONSTRUCT_COUNTS.csv", constructs.to_csv(index=False).encode("utf-8"))
    atomic_write(OUT / "ACCESS_LEDGER.json", (json.dumps(ledger, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    atomic_write(OUT / "RUN_STATUS.json", (json.dumps(status, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    atomic_write(OUT / "REPORT.md", report.encode("utf-8"))

    manifest_rows = []
    for path in [RUNNER, CONTRACT] + [OUT / name for name in OUTPUTS[:-1]]:
        manifest_rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    atomic_write(OUT / "RESULTS_SHA256.csv", manifest.to_csv(index=False).encode("utf-8"))
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
