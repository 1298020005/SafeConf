#!/usr/bin/env python3
"""Train/evaluate native PRESCRIBE on an E91 frozen Norman panel.

The runner imports the pinned upstream implementation instead of copying model or
loss code.  Its additions are experiment controls and durable raw/task outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
PRESCRIBE = Path("/home/yyf/archive/external/PRESCRIBE")
OUT_ROOT = ROOT / "docs/实验结果/E95_prescribe_norman_native_20260712"
COMMIT = "6f7264a205aaff654a9594863c5c10b656f88ebe"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", choices=["p1", "p2"], required=True)
    parser.add_argument("--mode", choices=["smoke", "formal"], required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--warmup-epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--limit-train-batches", type=int)
    parser.add_argument("--limit-val-batches", type=int)
    parser.add_argument("--limit-test-batches", type=int)
    return parser.parse_args()


def task_table(result: dict[str, np.ndarray], control_mean: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    labels = np.asarray(result["pert_cat"]).astype(str)
    for task in sorted(set(labels)):
        mask = labels == task
        pred = np.asarray(result["pred"])[mask]
        truth = np.asarray(result["truth"])[mask]
        pred_mean = pred.mean(axis=0)
        truth_mean = truth.mean(axis=0)
        pred_delta = pred_mean - control_mean
        truth_delta = truth_mean - control_mean
        rows.append(
            {
                "task_id": task,
                "n_cells": int(mask.sum()),
                "rmse_cell_gene": float(np.sqrt(np.mean((pred - truth) ** 2))),
                "rmse_mean_profile": float(np.sqrt(np.mean((pred_mean - truth_mean) ** 2))),
                "mae_cell_gene": float(np.mean(np.abs(pred - truth))),
                "magnitude_pred_rms": float(np.sqrt(np.mean(pred_delta**2))),
                "magnitude_truth_rms_diagnostic_only": float(np.sqrt(np.mean(truth_delta**2))),
                "epistemic_confidence": float(np.asarray(result["epistemic_conf"])[mask].mean()),
                "aleatoric_confidence": float(np.asarray(result["aleatoric_conf"])[mask].mean()),
                "combined_confidence_official": float(
                    2 * np.asarray(result["epistemic_conf"])[mask].mean()
                    + np.asarray(result["aleatoric_conf"])[mask].mean()
                ),
            }
        )
    table = pd.DataFrame(rows)
    for name in ["epistemic", "aleatoric", "combined"]:
        table[f"risk_{name}"] = -table[f"{name}_confidence" if name != "combined" else "combined_confidence_official"]
    return table


def summarize(table: pd.DataFrame) -> dict[str, object]:
    error = table["rmse_cell_gene"].to_numpy()
    out: dict[str, object] = {"n_tasks": int(len(table)), "n_cells": int(table["n_cells"].sum())}
    for score in ["risk_epistemic", "risk_aleatoric", "risk_combined", "magnitude_pred_rms"]:
        stat = spearmanr(table[score], error)
        out[f"rho_{score}_vs_error"] = float(stat.statistic)
        out[f"p_{score}_vs_error"] = float(stat.pvalue)
    curves: list[dict[str, object]] = []
    for score in ["risk_epistemic", "risk_aleatoric", "risk_combined", "magnitude_pred_rms"]:
        ordered = table.sort_values(score, ascending=True)
        for reject in [0.0, 0.1, 0.2, 0.3]:
            keep = max(1, int(np.floor(len(ordered) * (1 - reject))))
            curves.append(
                {
                    "score": score,
                    "reject_fraction": reject,
                    "n_retained": keep,
                    "mean_error_retained": float(ordered.iloc[:keep]["rmse_cell_gene"].mean()),
                    "high_error_recall_rejected": float(
                        len(
                            set(ordered.iloc[keep:]["task_id"])
                            & set(table.nlargest(max(1, len(table) - keep), "rmse_cell_gene")["task_id"])
                        )
                        / max(1, len(table) - keep)
                    ),
                }
            )
    out["selection_curve"] = curves
    return out


def main() -> None:
    cli = parse_args()
    defaults = {
        "smoke": {"max_epochs": 1, "warmup_epochs": 0, "limit_train_batches": 2, "limit_val_batches": 2, "limit_test_batches": 2},
        "formal": {"max_epochs": 50, "warmup_epochs": 5, "limit_train_batches": None, "limit_val_batches": None, "limit_test_batches": None},
    }[cli.mode]
    for key, value in defaults.items():
        if getattr(cli, key) is None:
            setattr(cli, key, value)

    run_name = f"norman_{cli.panel}_{cli.mode}_seed{cli.seed}"
    out = OUT_ROOT / run_name
    out.mkdir(parents=True, exist_ok=True)
    os.environ.update({"CUDA_LAUNCH_BLOCKING": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "WANDB_MODE": "offline"})
    os.chdir(PRESCRIBE)
    sys.path.insert(0, str(PRESCRIBE))
    from Step2_train import (  # noqa: PLC0415
        build_model,
        create_main_trainer_module,
        load_and_preprocess_data,
        run_warmup,
    )
    from src.model import EvidenceCallback, NaturalPosteriorNetworkLightningModule  # noqa: PLC0415

    seed_everything(cli.seed, workers=True)
    args = SimpleNamespace(
        seed=cli.seed,
        data_name=f"norman_{cli.panel}",
        backbone=None,
        batch_size=cli.batch_size,
        latent_dim=64,
        output_dim=10,
        flow_layers=10,
        flow_size=0.774,
        flow_n_hidden=2,
        maf_layers=10,
        budget="exp",
        bound=30,
        warmup_epochs=cli.warmup_epochs,
        warmup_lr=1e-3,
        log_prob_positive=False,
        accumulate_grad_batches=4,
        load_from="",
        lr=1e-4,
        lam1=1e-7,
        scheduler="plateau",
        interval="epoch",
        change_step=2,
        reduce_rate=0.99,
        warmup_steps=0,
        warmup_max_steps=int(875 * 1.25),
        lam2=0.1,
        lam3=1e-5,
    )
    status = {
        "experiment": "E95_prescribe_norman_native",
        "run_name": run_name,
        "mode": cli.mode,
        "phase": "started",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "prescribe_commit": COMMIT,
        "seed": cli.seed,
        "max_epochs": cli.max_epochs,
        "warmup_epochs": cli.warmup_epochs,
        "batch_size": cli.batch_size,
        "limit_train_batches": cli.limit_train_batches,
        "limit_val_batches": cli.limit_val_batches,
        "limit_test_batches": cli.limit_test_batches,
        "native_model_loss": True,
        "target_test_expression_used_for_task_selection": False,
    }
    (out / "STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")

    started = time.time()
    data_module, pert_data = load_and_preprocess_data(args)
    model = build_model(args, pert_data)
    logger = TensorBoardLogger(str(out / "logs"), name="prescribe")
    run_warmup(args, model, data_module, logger)
    module = create_main_trainer_module(args, model, pert_data)
    checkpoint = ModelCheckpoint(dirpath=str(out / "checkpoints"), monitor="val/loss", mode="min", save_top_k=1)
    evidence = EvidenceCallback(metrics_names=["pred", "truth", "pred_de", "truth_de"])
    trainer = Trainer(
        deterministic=True,
        callbacks=[checkpoint, evidence],
        max_epochs=cli.max_epochs,
        devices=1,
        accelerator="gpu",
        logger=logger,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        accumulate_grad_batches=args.accumulate_grad_batches,
        limit_train_batches=cli.limit_train_batches,
        limit_val_batches=cli.limit_val_batches,
        limit_test_batches=cli.limit_test_batches,
    )
    trainer.fit(model=module, datamodule=data_module)
    evaluated = module
    if checkpoint.best_model_path:
        try:
            evaluated = NaturalPosteriorNetworkLightningModule.load_from_checkpoint(checkpoint.best_model_path)
        except Exception as exc:  # preserve a usable run if the upstream checkpoint serializer fails
            status["checkpoint_reload_error"] = repr(exc)
    trainer.test(model=evaluated, datamodule=data_module)

    raw = evidence.result
    expected_tasks = set(pert_data.set2conditions["test"])
    observed_tasks = set(np.asarray(raw["pert_cat"]).astype(str))
    if cli.mode == "formal" and expected_tasks != observed_tasks:
        raise RuntimeError(
            "Frozen test-task contract mismatch: "
            f"missing={sorted(expected_tasks - observed_tasks)}, "
            f"unexpected={sorted(observed_tasks - expected_tasks)}"
        )
    np.savez_compressed(out / "test_predictions_raw.npz", **raw)
    data = pert_data.adata
    control_mean = np.asarray(data[data.obs["condition"] == "ctrl"].X.mean(axis=0)).reshape(-1)
    table = task_table(raw, control_mean)
    table.to_csv(out / "task_prediction_records.csv", index=False)
    summary = summarize(table)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    status.update(
        {
            "phase": "complete",
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "runtime_seconds": round(time.time() - started, 3),
            "best_checkpoint": checkpoint.best_model_path,
            "raw_prediction_sha256": sha256(out / "test_predictions_raw.npz"),
            "task_records_sha256": sha256(out / "task_prediction_records.csv"),
            "n_test_tasks_observed": int(len(table)),
            "n_test_cells_observed": int(table["n_cells"].sum()),
            "n_test_tasks_expected": int(len(expected_tasks)),
            "missing_test_tasks": sorted(expected_tasks - observed_tasks),
            "unexpected_test_tasks": sorted(observed_tasks - expected_tasks),
        }
    )
    (out / "STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "selection_curve"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
