#!/usr/bin/env python3
"""Build metadata-only Jiang24 assets for the time-bounded E216 confirmation.

The script never opens ``X`` or any expression layer.  It keeps every official
test row unchanged, deterministically caps train/validation cells within each
biological task, and exports the 4,000 highly-variable genes already annotated
in the published Jiang24 file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


EXPECTED_H5_BYTES = 93_532_364_449
EXPECTED_SPLIT_SHA256 = (
    "5af7da86a5b3994d570c0b1957d91f17cebb9f9b1943bac738b74fdb14b2ef5d"
)
EXPECTED_SHAPE = (1_628_476, 15_473)
EXPECTED_HVG = 4_000
HASH_NAMESPACE = "E216_JIANG24_RESOURCE_V1"


class AssetFailure(RuntimeError):
    """The frozen metadata, split, or selection contract failed."""


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def stable_priority(barcode: str) -> int:
    digest = hashlib.blake2b(
        f"{HASH_NAMESPACE}|{barcode}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def resource_split(
    metadata: pd.DataFrame,
    *,
    perturbed_cap: int = 16,
    control_cap: int = 128,
) -> pd.Series:
    """Return train/val caps while preserving the complete test split."""

    required = {"cell_barcode", "split", "cell_type", "treatment", "condition"}
    if not required.issubset(metadata.columns):
        raise AssetFailure(f"metadata columns missing: {sorted(required-set(metadata.columns))}")
    if metadata.cell_barcode.duplicated().any():
        raise AssetFailure("cell barcodes are not unique")
    if set(metadata.split) != {"train", "val", "test"}:
        raise AssetFailure("official split labels changed")
    if perturbed_cap < 1 or control_cap < perturbed_cap:
        raise AssetFailure("invalid train/validation cell caps")

    selected = metadata.split.astype(str).copy()
    selected.loc[selected.isin(["train", "val"])] = "drop"
    development = metadata.loc[metadata.split.isin(["train", "val"])].copy()
    development["priority"] = np.fromiter(
        (stable_priority(value) for value in development.cell_barcode.astype(str)),
        dtype=np.uint64,
        count=len(development),
    )
    group_columns = ["split", "cell_type", "treatment", "condition"]
    for keys, block in development.groupby(group_columns, sort=True, observed=True):
        cap = control_cap if str(keys[-1]) == "control" else perturbed_cap
        keep = block.nsmallest(min(cap, len(block)), "priority").index
        selected.loc[keep] = str(keys[0])

    if not selected.loc[metadata.split.eq("test")].eq("test").all():
        raise AssetFailure("official test membership changed")
    if selected.loc[metadata.split.isin(["train", "val"])].eq("test").any():
        raise AssetFailure("development row was promoted into test")
    return selected


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--inventory-status", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--perturbed-cap", type=int, default=16)
    parser.add_argument("--control-cap", type=int, default=128)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    import anndata as ad

    h5ad_path = args.h5ad.resolve()
    split_path = args.split.resolve()
    inventory_path = args.inventory_status.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise AssetFailure(f"refusing to overwrite: {output_dir}")
    if not h5ad_path.is_file() or h5ad_path.stat().st_size != EXPECTED_H5_BYTES:
        raise AssetFailure("Jiang24 H5 identity changed")
    if sha256_file(split_path) != EXPECTED_SPLIT_SHA256:
        raise AssetFailure("official Jiang24 split changed")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if (
        inventory.get("status") != "PASS"
        or inventory.get("expression_values_read") is not False
        or inventory.get("target_truth_used_for_selection") is not False
    ):
        raise AssetFailure("E208 metadata inventory gate is incomplete")

    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        if (adata.n_obs, adata.n_vars) != EXPECTED_SHAPE:
            raise AssetFailure("Jiang24 matrix shape changed")
        required = ["cell_type", "treatment", "condition"]
        if not set(required).issubset(adata.obs.columns):
            raise AssetFailure("Jiang24 task metadata changed")
        metadata = adata.obs.loc[:, required].copy()
        metadata.index = metadata.index.astype(str)
        hvg = adata.var.loc[adata.var["highly_variable"].astype(bool)].index.astype(str)
    finally:
        adata.file.close()
    if len(hvg) != EXPECTED_HVG or pd.Index(hvg).has_duplicates:
        raise AssetFailure("published 4,000-HVG annotation changed")

    official = pd.read_csv(
        split_path, header=None, names=["cell_barcode", "split"], dtype=str
    ).set_index("cell_barcode")
    if not official.index.equals(metadata.index):
        raise AssetFailure("official split order no longer matches H5 observations")
    metadata["split"] = official["split"].to_numpy()
    metadata = metadata.reset_index(names="cell_barcode")
    for column in ("cell_barcode", "cell_type", "treatment", "condition", "split"):
        metadata[column] = metadata[column].astype(str)

    selected = resource_split(
        metadata,
        perturbed_cap=args.perturbed_cap,
        control_cap=args.control_cap,
    )
    output_dir.mkdir(parents=True)
    derived_split = output_dir / "jiang24_e216_resource_split.csv"
    hvg_path = output_dir / "jiang24_official_hvg4000.csv"
    split_frame = pd.DataFrame(
        {"cell_barcode": metadata.cell_barcode, "split": selected}
    )
    atomic_text(derived_split, split_frame.to_csv(index=False, header=False))
    atomic_text(hvg_path, pd.Series(hvg).to_csv(index=False, header=False))

    group_columns = ["split", "cell_type", "treatment", "condition"]
    retained = metadata.assign(resource_split=selected)
    development = retained.loc[retained.resource_split.isin(["train", "val"])]
    cap_check = (
        development.groupby(group_columns, observed=True, sort=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    perturbed_ok = cap_check.loc[cap_check.condition.ne("control"), "n_cells"].le(
        args.perturbed_cap
    ).all()
    control_ok = cap_check.loc[cap_check.condition.eq("control"), "n_cells"].le(
        args.control_cap
    ).all()
    original_counts = metadata.split.value_counts().to_dict()
    selected_counts = selected.value_counts().to_dict()
    status = {
        "experiment": "E216_jiang24_resource_bounded_confirmation",
        "stage": "D0_METADATA_ONLY_RESOURCE_ASSETS",
        "status": "PASS" if perturbed_ok and control_ok else "FAIL",
        "created_at": now(),
        "selection_namespace": HASH_NAMESPACE,
        "perturbed_cells_per_development_task_cap": args.perturbed_cap,
        "control_cells_per_development_state_cap": args.control_cap,
        "feature_panel": "published_jiang24_highly_variable",
        "n_features": len(hvg),
        "official_split_counts": {str(k): int(v) for k, v in original_counts.items()},
        "resource_split_counts": {str(k): int(v) for k, v in selected_counts.items()},
        "n_test_rows_preserved": int(selected.eq("test").sum()),
        "test_membership_unchanged": bool(
            selected.eq("test").equals(metadata.split.eq("test"))
        ),
        "expression_values_read": False,
        "test_perturbed_expression_rows_read": 0,
        "target_truth_used_for_selection": False,
        "files": {
            derived_split.name: {
                "bytes": derived_split.stat().st_size,
                "sha256": sha256_file(derived_split),
            },
            hvg_path.name: {
                "bytes": hvg_path.stat().st_size,
                "sha256": sha256_file(hvg_path),
            },
        },
    }
    atomic_text(
        output_dir / "E216_RESOURCE_ASSET_STATUS.json",
        json.dumps(status, ensure_ascii=False, indent=2) + "\n",
    )
    return status


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
