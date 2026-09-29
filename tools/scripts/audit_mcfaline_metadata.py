#!/usr/bin/env python3
"""Inspect McFaline metadata without reading the expression matrix.

This program deliberately uses backed AnnData and only copies ``obs``/``var``
metadata.  It never touches ``adata.X`` or a layer containing expression truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import pandas as pd


DEFAULT_DATA = Path("/home/yyf/data/perturbench_mcfaline23_official/mcfaline23_gxe_processed.h5ad")
DEFAULT_SPLIT = Path("/home/yyf/data/perturbench_mcfaline23_official/splits/mcfaline23_gxe_splits/full_covariate_split.csv")
DEFAULT_OUT = Path(
    "/home/yyf/runtime_worktrees/e220_reviewer_closure_20260921/"
    "docs/实验结果/Stage2_mature_upstream_20260928/external_asset_audit"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def task_frame(obs: pd.DataFrame, split: pd.Series) -> pd.DataFrame:
    required = ["condition", "cell_type", "treatment"]
    missing = [name for name in required if name not in obs]
    if missing:
        raise ValueError(f"required metadata columns missing: {missing}")
    frame = obs[required].copy()
    frame["split"] = split.reindex(frame.index)
    if frame["split"].isna().any():
        raise ValueError(f"{int(frame['split'].isna().sum())} cells missing from split manifest")
    frame["is_control"] = frame["condition"].astype(str).eq("control")
    frame["state_id"] = frame["cell_type"].astype(str) + "::" + frame["treatment"].astype(str)
    frame["task_id"] = frame["state_id"] + "::" + frame["condition"].astype(str)
    return frame


def summarize_tasks(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    task_counts = (
        frame.groupby(["split", "state_id", "cell_type", "treatment", "condition", "is_control"], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    for split_name, part in frame.groupby("split", observed=True):
        treated = part.loc[~part["is_control"]]
        rows.append(
            {
                "split": split_name,
                "n_cells": len(part),
                "n_states": part["state_id"].nunique(),
                "n_perturbation_clusters": treated["condition"].nunique(),
                "n_treated_tasks": treated["task_id"].nunique(),
                "n_control_tasks": part.loc[part["is_control"], "state_id"].nunique(),
            }
        )
    return task_counts, pd.DataFrame(rows).sort_values("split")


def history_eligibility(task_counts: pd.DataFrame) -> pd.DataFrame:
    train = task_counts[(task_counts["split"] == "train") & (~task_counts["is_control"])].copy()
    test = task_counts[(task_counts["split"] == "test") & (~task_counts["is_control"])].copy()
    train_by_pert = train.groupby("condition", observed=True).agg(
        train_history_cells=("n_cells", "sum"),
        train_history_states=("state_id", "nunique"),
    )
    test = test.merge(train_by_pert, how="left", left_on="condition", right_index=True)
    test[["train_history_cells", "train_history_states"]] = test[
        ["train_history_cells", "train_history_states"]
    ].fillna(0).astype(int)
    test["has_internal_history"] = test["train_history_cells"] > 0
    test["history_is_cross_state"] = test["train_history_states"] > 0
    # This file is an eligibility inventory only. No expression-derived content
    # or current test response is read or written.
    return test[
        [
            "state_id", "cell_type", "treatment", "condition", "n_cells",
            "train_history_cells", "train_history_states", "has_internal_history",
            "history_is_cross_state",
        ]
    ].sort_values(["state_id", "condition"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    split = pd.read_csv(args.split, header=None, names=["cell_id", "split"], index_col="cell_id")["split"]
    data = ad.read_h5ad(args.data, backed="r")
    try:
        obs = data.obs.copy()
        var_names = data.var_names.astype(str).copy()
        shape = tuple(map(int, data.shape))
    finally:
        data.file.close()

    frame = task_frame(obs, split)
    task_counts, split_summary = summarize_tasks(frame)
    eligible = history_eligibility(task_counts)
    task_counts.to_csv(args.out / "MCFALINE_TASK_METADATA.csv", index=False)
    split_summary.to_csv(args.out / "MCFALINE_SPLIT_SUMMARY.csv", index=False)
    eligible.to_csv(args.out / "MCFALINE_TEST_HISTORY_ELIGIBILITY.csv", index=False)

    summary = {
        "audit_time_utc": datetime.now(timezone.utc).isoformat(),
        "metadata_only": True,
        "expression_matrix_read": False,
        "test_truth_read": False,
        "data_path": str(args.data),
        "data_sha256": sha256(args.data),
        "split_path": str(args.split),
        "split_sha256": sha256(args.split),
        "shape": {"cells": shape[0], "genes": shape[1]},
        "obs_columns": list(map(str, obs.columns)),
        "gene_id_count": len(var_names),
        "gene_id_unique": len(set(var_names)) == len(var_names),
        "split_summary": split_summary.to_dict(orient="records"),
        "test_history": {
            "n_test_tasks": int(len(eligible)),
            "n_with_internal_history": int(eligible["has_internal_history"].sum()),
            "coverage": float(eligible["has_internal_history"].mean()) if len(eligible) else None,
            "median_train_history_cells": float(eligible["train_history_cells"].median()) if len(eligible) else None,
            "median_train_history_states": float(eligible["train_history_states"].median()) if len(eligible) else None,
        },
    }
    (args.out / "MCFALINE_METADATA_AUDIT.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
