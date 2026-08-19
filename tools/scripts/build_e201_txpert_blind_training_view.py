#!/usr/bin/env python3
"""Build a target-truth-free TxPert training cache for E201."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import anndata as ad
import joblib
import pandas as pd


PUBLIC_CELL_TYPES = ("K562", "RPE1", "hepg2", "jurkat")
EXPECTED_INPUT_SHA256 = (
    "1b557390148eba358304e43e0b239538d9ae0691b26ec843f41cf544960307a8"
)


class ViewFailure(RuntimeError):
    pass


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--target", choices=PUBLIC_CELL_TYPES, required=True)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    source = args.source_cache.resolve()
    output = args.output_cache.resolve()
    dataset = source / "de_adata_test.h5ad"
    split_path = source / "splits/train_test_split.pkl"
    subgroup_path = source / "splits/subgroup.pkl"
    for path in (dataset, split_path, subgroup_path):
        if not path.is_file():
            raise ViewFailure(f"missing input: {path}")
    if output.exists():
        raise ViewFailure(f"refusing to overwrite: {output}")
    input_sha = sha256_file(dataset)
    if input_sha != EXPECTED_INPUT_SHA256:
        raise ViewFailure("official H5AD hash mismatch")

    split = joblib.load(split_path)
    train_conditions = set(map(str, split["train"]))
    val_conditions = set(map(str, split["val"]))
    allowed_conditions = train_conditions | val_conditions
    source_contexts = sorted(set(PUBLIC_CELL_TYPES) - {args.target})

    backed = ad.read_h5ad(dataset, backed="r")
    obs = backed.obs.copy()
    is_control = obs.control.astype(bool)
    allowed_control = is_control & obs.cell_line.isin(PUBLIC_CELL_TYPES)
    allowed_treatment = (
        ~is_control
        & obs.cell_line.isin(source_contexts)
        & obs.condition.astype(str).isin(allowed_conditions)
    )
    keep = (allowed_control | allowed_treatment).to_numpy(bool)
    inventory_before = (
        obs.assign(control_bool=is_control)
        .groupby(["cell_line", "control_bool"], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    subset = backed[keep].to_memory()
    backed.file.close()
    subset.uns.clear()
    out_obs = subset.obs.copy()
    n_genes = int(subset.n_vars)
    out_control = out_obs.control.astype(bool)
    target_treatment = int((~out_control & out_obs.cell_line.eq(args.target)).sum())
    unexpected_contexts = sorted(set(out_obs.cell_line.astype(str)) - set(PUBLIC_CELL_TYPES))
    unexpected_treatments = sorted(
        set(out_obs.loc[~out_control, "cell_line"].astype(str)) - set(source_contexts)
    )
    forbidden_conditions = int(
        (
            ~out_control
            & ~out_obs.condition.astype(str).isin(allowed_conditions)
        ).sum()
    )
    if target_treatment != 0:
        raise ViewFailure("target perturbed cells survived blind view")
    if unexpected_contexts or unexpected_treatments or forbidden_conditions:
        raise ViewFailure("blind view contract failed")

    output.mkdir(parents=True)
    (output / "splits").mkdir()
    output_dataset = output / "de_adata_test.h5ad"
    subset.write_h5ad(output_dataset, compression="gzip")
    shutil.copy2(split_path, output / "splits/train_test_split.pkl")
    shutil.copy2(subgroup_path, output / "splits/subgroup.pkl")

    inventory_after = (
        out_obs.assign(control_bool=out_control)
        .groupby(["cell_line", "control_bool"], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    inventory_before.to_csv(output / "E201_SOURCE_CONTEXT_INVENTORY.csv", index=False)
    inventory_after.to_csv(output / "E201_BLIND_CONTEXT_INVENTORY.csv", index=False)
    del subset

    files = []
    for role, path in (
        ("blind_training_h5ad", output_dataset),
        ("condition_split", output / "splits/train_test_split.pkl"),
        ("subgroup_labels", output / "splits/subgroup.pkl"),
        ("source_inventory", output / "E201_SOURCE_CONTEXT_INVENTORY.csv"),
        ("blind_inventory", output / "E201_BLIND_CONTEXT_INVENTORY.csv"),
    ):
        files.append(
            {
                "role": role,
                "path": path.name if path.parent == output else path.relative_to(output).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "experiment": "E201_txpert_multitarget_retraining",
        "status": "BLIND_TRAINING_VIEW_READY",
        "created_at": now(),
        "target": args.target,
        "source_contexts": source_contexts,
        "source_h5ad": str(dataset),
        "source_h5ad_bytes": dataset.stat().st_size,
        "source_h5ad_sha256": input_sha,
        "n_rows": int(len(out_obs)),
        "n_genes": n_genes,
        "n_controls": int(out_control.sum()),
        "n_target_controls": int(
            (out_control & out_obs.cell_line.eq(args.target)).sum()
        ),
        "n_source_treatments": int((~out_control).sum()),
        "n_target_treatments": target_treatment,
        "uns_keys": [],
        "k562_adamson_rows": int(out_obs.cell_line.eq("K562_adamson").sum()),
        "files": files,
    }
    write_json(output / "E201_BLIND_VIEW_MANIFEST.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
