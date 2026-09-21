#!/usr/bin/env python3
"""Freeze the E208 counterfactual prediction manifest from the D0 inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd


class ManifestFailure(RuntimeError):
    """The D0 inventory no longer satisfies the frozen E208 contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    args = parser.parse_args()

    inventory_path = args.inventory.expanduser().absolute()
    inventory = pd.read_csv(inventory_path)
    required = {
        "split",
        "cell_type",
        "treatment",
        "condition",
        "is_control",
        "is_single_gene",
        "is_primary_context",
        "passes_ge30",
    }
    if not required.issubset(inventory.columns):
        raise ManifestFailure(f"inventory fields missing: {sorted(required-inventory.columns)}")
    selected = inventory.loc[
        inventory["split"].eq("test")
        & inventory["is_primary_context"].astype(bool)
        & inventory["is_single_gene"].astype(bool)
        & ~inventory["is_control"].astype(bool)
        & inventory["passes_ge30"].astype(bool)
    ].copy()
    selected["task_id"] = (
        selected.cell_type.astype(str)
        + "|"
        + selected.treatment.astype(str)
        + "::"
        + selected.condition.astype(str)
    )
    selected = selected.sort_values(
        ["cell_type", "treatment", "condition"], kind="mergesort"
    ).reset_index(drop=True)
    contexts = selected[["cell_type", "treatment"]].drop_duplicates()
    if len(selected) != 224 or selected.task_id.nunique() != 224 or len(contexts) != 12:
        raise ManifestFailure(
            f"expected 224 unique tasks in 12 contexts, observed "
            f"{len(selected)}, {selected.task_id.nunique()}, {len(contexts)}"
        )
    manifest = selected[["condition", "cell_type", "treatment", "task_id"]]
    atomic_csv(args.output, manifest)
    status = {
        "experiment": "E208_jiang24_external_confirmation",
        "stage": "D1_PRETRUTH_PREDICTION_MANIFEST",
        "status": "PASS",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "inventory_path": str(inventory_path),
        "inventory_sha256": sha256_file(inventory_path),
        "manifest_path": str(args.output.expanduser().absolute()),
        "manifest_sha256": sha256_file(args.output),
        "n_tasks": len(manifest),
        "n_contexts": len(contexts),
        "context_task_counts": {
            f"{cell_type}|{treatment}": int(count)
            for (cell_type, treatment), count in manifest.groupby(
                ["cell_type", "treatment"], sort=True
            ).size().items()
        },
        "test_perturbed_expression_rows_read": 0,
        "target_truth_access": "NOT_AUTHORIZED",
    }
    atomic_json(args.status, status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
