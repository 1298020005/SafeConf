#!/usr/bin/env python3
"""Run one E204 source-only weighted TxPert training job.

The E201 adapter remains frozen.  This launcher applies a narrowly scoped
runtime patch: it attaches a condition-level source-only weight to each
training sample and replaces the scalar reconstruction loss with its weighted
per-sample equivalent.  The external TxPert checkout is never modified.
"""

from __future__ import annotations

import argparse
import json
import runpy
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--txpert-repo", type=Path, required=True)
    p.add_argument("--task-type", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--kind", choices=("smoke", "profile", "formal"), required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--smoke-train-batches", type=int, default=20)
    p.add_argument("--task-weight-manifest", type=Path, required=True)
    p.add_argument("--weight-column", choices=("task_weight", "dispersion_only_weight"), default="task_weight")
    return p.parse_args()


def load_weights(path: Path, target: str, column: str) -> dict[str, float]:
    frame = pd.read_csv(path)
    required = {"target", "condition", column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"weight manifest missing columns: {missing}")
    view = frame[frame["target"].astype(str) == target].copy()
    if view.empty:
        raise SystemExit(f"weight manifest has no rows for target {target}")
    if view["condition"].duplicated().any():
        raise SystemExit(f"duplicate conditions for target {target}")
    values = pd.to_numeric(view[column], errors="raise")
    if not values.map(pd.notna).all() or (values <= 0).any():
        raise SystemExit("weights must be finite and positive")
    return dict(zip(view["condition"].astype(str), values.astype(float)))


def main() -> None:
    args = parse_args()
    manifest = args.task_weight_manifest.resolve()
    weights = load_weights(manifest, args.target, args.weight_column)

    # The frozen E201 script imports these modules after runpy starts.  Patch
    # their class/function objects before it constructs the datamodule.
    sys.path.insert(0, str(args.txpert_repo.resolve()))
    import gspp.data.data_utils as data_utils
    import gspp.data.datamodule as datamodule_module
    from gspp.predictor import PertPredictor

    original_dataset_getitem = datamodule_module.XcelltypePerturbDataset.__getitem__
    unknown_conditions: dict[str, int] = {}

    def condition_weight(condition: str, perturbations) -> float | None:
        if condition in weights:
            return weights[condition]
        # TxPert appends cell-line and replicate suffixes to condition names,
        # e.g. ``RPE1_DDX5+ctrl_1+1``.  Match only the manifest condition
        # prefix; the suffix is an experimental label, not a new task.
        tail = condition.split("_", 1)[1] if "_" in condition else condition
        for key, value in weights.items():
            if tail == key or tail.startswith(key + "_"):
                return value
        is_control = condition.lower() in {"ctrl", "control"} or perturbations in (
            [-1],
            ["ctrl"],
        ) or tail.lower().startswith("ctrl")
        return 1.0 if is_control else None

    def weighted_getitem(self, idx):
        item = original_dataset_getitem(self, idx)
        condition = str(item[4])
        perturbations = item[3]
        # The adapter appends matched controls to the training dataset;
        # controls are reconstruction anchors and retain unit weight.
        weight = condition_weight(condition, perturbations)
        if weight is None:
            # Conditions absent from the strict E201 target task table are
            # still valid source training examples.  They receive unit weight
            # and are counted for coverage auditing; they are never silently
            # assigned a guessed difficulty.
            unknown_conditions[condition] = unknown_conditions.get(condition, 0) + 1
            weight = 1.0
        return (*item, float(weight))

    datamodule_module.XcelltypePerturbDataset.__getitem__ = weighted_getitem

    def weighted_collate(batch):
        base = data_utils.collate_fn([row[:9] for row in batch])
        base.task_weights = torch.tensor(
            [row[9] for row in batch], dtype=torch.float32
        )
        return base

    # PertDataModule resolves collate_fn from its own module namespace.
    datamodule_module.collate_fn = weighted_collate

    def weighted_training_step(self, batch, batch_idx):
        prediction, intrinsic_mean, intrinsic_log_var = self.forward(
            batch.control, batch.pert_idxs, batch.p
        )
        target = batch.x[:, : self.adata_output_dim]
        weights_tensor = batch.task_weights.to(prediction.device)
        mse = (prediction - target).pow(2).mean(dim=1)
        cosine = 1.0 - F.cosine_similarity(prediction, target, dim=-1)
        mse_weight = float(getattr(self.model, "mse_weight", 1.0))
        per_sample = mse_weight * mse + (1.0 - mse_weight) * cosine
        recon_loss = (per_sample * weights_tensor).sum() / weights_tensor.sum()
        kl = 0.0
        if not getattr(self.model, "no_basal_model", True):
            if getattr(self.model, "cntr_model_type", None) == "vae":
                kl = self.model.kl_divergence(intrinsic_mean, intrinsic_log_var)
        loss = recon_loss + kl
        self.log("weighted_recon_loss", recon_loss, prog_bar=True, logger=True)
        self.log("mean_task_weight", weights_tensor.mean(), logger=True)
        return loss

    PertPredictor.training_step = weighted_training_step

    # Present exactly the arguments expected by the frozen adapter.
    sys.argv = [
        "/home/yyf/proj/tools/scripts/txpert_blind_training_adapter.py",
        "--txpert-repo",
        str(args.txpert_repo),
        "--task-type",
        args.task_type,
        "--target",
        args.target,
        "--seed",
        str(args.seed),
        "--run-dir",
        str(args.run_dir),
        "--kind",
        args.kind,
        "--batch-size",
        str(args.batch_size),
        "--smoke-train-batches",
        str(args.smoke_train_batches),
    ]
    status = "FAILED"
    try:
        runpy.run_path(
            "/home/yyf/proj/tools/scripts/txpert_blind_training_adapter.py",
            run_name="__main__",
        )
        status = "COMPLETE"
    finally:
        run_dir = args.run_dir.resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest, run_dir / "E204_SOURCE_ONLY_TASK_WEIGHTS.csv")
        metadata = {
            "experiment": "E204_risk_guided_training",
            "status": status,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "target": args.target,
            "seed": args.seed,
            "kind": args.kind,
            "weight_column": args.weight_column,
            "weight_manifest": str(manifest),
            "weight_manifest_rows_for_target": len(weights),
            "unit_weight_fallback_condition_counts": unknown_conditions,
            "unit_weight_fallback_samples": int(sum(unknown_conditions.values())),
            "target_expression_opened": False,
            "external_txpert_modified": False,
            "loss": "weighted per-condition TxPert reconstruction loss; controls have unit weight",
        }
        (run_dir / "E204_WEIGHTING_STATUS.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
