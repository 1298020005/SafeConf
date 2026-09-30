#!/usr/bin/env python3
"""Build a metadata-only McFaline prediction manifest for the sealed test split.

The script opens the AnnData file in backed read-only mode and reads only ``obs``.
It never indexes or materialises ``X``.  The resulting CSV is suitable for
PerturBench's counterfactual ``predict`` entrypoint, which consumes task metadata
and matched control cells without attaching treated reference expression.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split = pd.read_csv(
        args.split, header=None, names=["cell_id", "split"], index_col="cell_id"
    )
    adata = ad.read_h5ad(args.h5ad, backed="r")
    try:
        if not adata.obs_names.equals(split.index):
            raise RuntimeError("split cell order/index does not match AnnData obs")
        # Metadata only: no access to adata.X is permitted in this script.
        obs = adata.obs.loc[
            split.index[split["split"].eq("test")],
            ["condition", "cell_type", "treatment", "perturbation", "control"],
        ].copy()
    finally:
        adata.file.close()

    treated = obs.loc[
        (~obs["condition"].astype(str).eq("control"))
        & (~obs["control"].astype(str).eq("1"))
    ]
    prediction = (
        treated[["condition", "cell_type", "treatment"]]
        .astype(str)
        .drop_duplicates()
        .sort_values(["condition", "cell_type", "treatment"], kind="mergesort")
        .reset_index(drop=True)
    )
    prediction.insert(
        0,
        "biological_task_key",
        prediction["condition"]
        + "|"
        + prediction["cell_type"]
        + "|"
        + prediction["treatment"],
    )
    prediction_path = args.output_dir / "TEST_PREDICTION_TASKS.csv"
    # PerturBench consumes the final three columns; the keyed file is retained
    # for SafeConf provenance and a key-free copy is used for prediction.
    prediction.to_csv(prediction_path, index=False)
    input_path = args.output_dir / "PERTURBENCH_TEST_PREDICTION_INPUT.csv"
    prediction[["condition", "cell_type", "treatment"]].to_csv(input_path, index=False)

    status = {
        "schema": "SafeConf-McFaline-Test-Prediction-Manifest-v1",
        "n_test_cells_metadata": int(len(obs)),
        "n_test_treated_cells_metadata": int(len(treated)),
        "n_test_tasks": int(len(prediction)),
        "n_perturbation_clusters": int(prediction["condition"].nunique()),
        "n_strata": int(
            prediction[["cell_type", "treatment"]].drop_duplicates().shape[0]
        ),
        "h5ad_expression_accessed": False,
        "test_treated_expression_accessed": False,
        "metadata_columns_read": [
            "condition",
            "cell_type",
            "treatment",
            "perturbation",
            "control",
        ],
        "split_sha256": sha256(args.split),
        "task_manifest_sha256": sha256(prediction_path),
        "prediction_input_sha256": sha256(input_path),
    }
    (args.output_dir / "RUN_STATUS.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
