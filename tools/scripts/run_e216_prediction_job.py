#!/usr/bin/env python3
"""Generate truth-free E216 counterfactual predictions from a frozen run.

The job only asks PerturBench to load unperturbed control cells at prediction
time.  Perturbed test expression is neither passed to the data iterator nor
opened by this script.  Raw predictions are deliberately retained so that a
later sealing stage can hash them before outcome evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from omegaconf import OmegaConf, open_dict


REQUIRED_TASK_COLUMNS = ("condition", "cell_type", "treatment")
PREDICTION_CONTROL_SEED = 216


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_prediction_frame(task_path: Path, *, require_224: bool) -> pd.DataFrame:
    frame = pd.read_csv(task_path)
    missing = [name for name in REQUIRED_TASK_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"task inventory misses columns: {missing}")
    frame = frame.loc[:, REQUIRED_TASK_COLUMNS].copy()
    if frame.isna().any().any():
        raise ValueError("prediction task keys must not contain missing values")
    if frame.duplicated(list(REQUIRED_TASK_COLUMNS)).any():
        raise ValueError("prediction task keys must be unique")
    if require_224 and len(frame) != 224:
        raise ValueError(f"formal E216 requires exactly 224 tasks, found {len(frame)}")
    return frame.sort_values(list(REQUIRED_TASK_COLUMNS), kind="stable").reset_index(drop=True)


def choose_checkpoint(run_dir: Path, status: dict, checkpoint: str | None) -> Path:
    if checkpoint:
        result = Path(checkpoint).resolve()
    else:
        candidate = status.get("best_checkpoint") or status.get("last_checkpoint")
        if not candidate:
            raise ValueError("training status does not name a completed checkpoint")
        result = Path(candidate).resolve()
    if not result.is_file():
        raise FileNotFoundError(result)
    if run_dir.resolve() not in result.parents:
        raise ValueError("checkpoint must belong to the declared training run")
    return result


def build_prediction_config(
    training_config_path: Path,
    *,
    checkpoint: Path,
    task_frame_path: Path,
    output_dir: Path,
    control_asset: Path,
    seed: int,
    max_control_cells: int,
    chunk_size: int,
    accelerator: str,
):
    cfg = OmegaConf.load(training_config_path)
    with open_dict(cfg):
        # These training-only sections contain `${hydra:runtime.output_dir}`
        # resolvers, which are unavailable in this standalone prediction job.
        # They are not consumed by PerturBench.predict and are removed before
        # the resolved prediction contract is serialized.
        if "callbacks" in cfg:
            del cfg.callbacks
        if "logger" in cfg:
            del cfg.logger
        cfg.ckpt_path = str(checkpoint)
        cfg.seed = int(seed)
        cfg.output_path = str(output_dir)
        cfg.paths.output_dir = str(output_dir)
        cfg.paths.work_dir = str(output_dir)
        cfg.task_name = "e216_truth_free_prediction"
        cfg.data.prediction = OmegaConf.create(
            {
                "prediction_dataframe_path": str(task_frame_path),
                "chunk_size": int(chunk_size),
                "predict_data_iter_factory": {
                    "_target_": "safeconf_audit.e216_prediction.counterfactual_from_fixed_controls",
                    "_partial_": True,
                    "fixed_control_h5": str(control_asset),
                    "max_control_cells_per_covariate": int(max_control_cells),
                    "perturbation_key": "condition",
                    "perturbation_control_value": "control",
                    "covariate_keys": ["cell_type", "treatment"],
                    "embedding_key": None,
                    "seed": PREDICTION_CONTROL_SEED,
                    "feature_filter_path": str(
                        cfg.data.data_iter_factory.feature_filter_path
                    ),
                },
            }
        )
        # Prediction does not need training callbacks or loggers.  A minimal
        # deterministic single-GPU trainer avoids Hydra runtime resolvers from
        # the original training job.
        cfg.trainer = OmegaConf.create(
            {
                "_target_": "lightning.pytorch.trainer.Trainer",
                "accelerator": accelerator,
                "devices": 1,
                "precision": "16-mixed" if accelerator == "gpu" else "32-true",
                "deterministic": True,
                "enable_checkpointing": False,
                "enable_progress_bar": True,
                "logger": False,
            }
        )
    return cfg


def aggregate_prediction_chunks(
    paths: list[Path],
    expected_tasks: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Aggregate cell predictions to one centroid per registered task."""

    sums: dict[tuple[str, str, str], np.ndarray] = {}
    counts: dict[tuple[str, str, str], int] = {}
    gene_names: np.ndarray | None = None
    for path in paths:
        current = ad.read_h5ad(path)
        current_genes = current.var_names.astype(str).to_numpy()
        if gene_names is None:
            gene_names = current_genes
        elif not np.array_equal(gene_names, current_genes):
            raise ValueError("prediction chunks do not share the frozen gene axis")
        values = current.X.toarray() if hasattr(current.X, "toarray") else np.asarray(current.X)
        values = np.asarray(values, dtype=np.float64)
        obs = current.obs
        for columns, positions in obs.groupby(
            list(REQUIRED_TASK_COLUMNS), observed=True, sort=False
        ).indices.items():
            key = tuple(str(value) for value in columns)
            block = values[np.asarray(positions, dtype=int)]
            block_sum = block.sum(axis=0, dtype=np.float64)
            sums[key] = sums.get(key, np.zeros_like(block_sum)) + block_sum
            counts[key] = counts.get(key, 0) + int(block.shape[0])
    if gene_names is None:
        raise ValueError("no prediction chunks were supplied")

    ordered_keys = [
        tuple(str(row[name]) for name in REQUIRED_TASK_COLUMNS)
        for _, row in expected_tasks.iterrows()
    ]
    missing = [key for key in ordered_keys if key not in sums]
    extra = sorted(set(sums) - set(ordered_keys))
    if missing or extra:
        raise ValueError(f"prediction task mismatch: missing={missing[:3]}, extra={extra[:3]}")
    centroids = np.stack([sums[key] / counts[key] for key in ordered_keys]).astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        predictions=centroids,
        gene_names=np.asarray(gene_names, dtype=str),
        condition=np.asarray([key[0] for key in ordered_keys], dtype=str),
        cell_type=np.asarray([key[1] for key in ordered_keys], dtype=str),
        treatment=np.asarray([key[2] for key in ordered_keys], dtype=str),
        n_control_predictions=np.asarray([counts[key] for key in ordered_keys], dtype=np.int32),
    )
    summary = expected_tasks.copy()
    summary["n_control_predictions"] = [counts[key] for key in ordered_keys]
    return summary


def build_control_centroids(cfg, task_frame: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Recreate the registered control draw and save state centroids only."""

    from perturbench.data.datasets import Counterfactual

    dataset, _ = Counterfactual.from_h5(
        str(cfg.data.prediction.predict_data_iter_factory.fixed_control_h5),
        task_frame.copy(),
        perturbation_key="condition",
        perturbation_control_value="control",
        covariate_keys=["cell_type", "treatment"],
        embedding_key=None,
        seed=PREDICTION_CONTROL_SEED,
        max_control_cells_per_covariate=128,
        feature_filter_path=str(cfg.data.data_iter_factory.feature_filter_path),
    )
    control = dataset.fetch_control_anndata()
    values = control.X.toarray() if hasattr(control.X, "toarray") else np.asarray(control.X)
    values = np.asarray(values, dtype=np.float64)
    keys = []
    centroids = []
    counts = []
    for key, positions in control.obs.groupby(
        ["cell_type", "treatment"], observed=True, sort=True
    ).indices.items():
        positions = np.asarray(positions, dtype=int)
        keys.append(tuple(str(value) for value in key))
        centroids.append(values[positions].mean(axis=0))
        counts.append(len(positions))
    np.savez_compressed(
        output_path,
        control_centroids=np.asarray(centroids, dtype=np.float32),
        gene_names=np.asarray(control.var_names.astype(str), dtype=str),
        cell_type=np.asarray([key[0] for key in keys], dtype=str),
        treatment=np.asarray([key[1] for key in keys], dtype=str),
        n_controls=np.asarray(counts, dtype=np.int32),
    )
    return pd.DataFrame(
        {
            "cell_type": [key[0] for key in keys],
            "treatment": [key[1] for key in keys],
            "n_controls": counts,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--perturbench-repo", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--control-asset", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--max-control-cells", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--accelerator", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--task-limit", type=int)
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--config-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_control_cells != 128:
        raise ValueError("E216 contract fixes max-control-cells=128")
    if args.chunk_size < 1:
        raise ValueError("chunk-size must be positive")
    if not args.allow_smoke and args.accelerator != "gpu":
        raise ValueError("formal E216 prediction is frozen to GPU inference")

    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    task_path = Path(args.tasks).resolve()
    control_asset = Path(args.control_asset).resolve()
    if not control_asset.is_file():
        raise FileNotFoundError(control_asset)
    training_config_path = run_dir / ".hydra" / "config.yaml"
    training_status_path = run_dir / "E216_RUN_STATUS.json"
    if not training_config_path.is_file() or not training_status_path.is_file():
        raise FileNotFoundError("run is missing resolved config or E216 status")
    status = json.loads(training_status_path.read_text(encoding="utf-8"))
    allowed_status = {"PASS"} if args.allow_smoke else {"COMPLETE"}
    if status.get("status") not in allowed_status:
        raise RuntimeError(
            f"training run must be {sorted(allowed_status)}, found {status.get('status')}"
        )
    mode = status.get("mode")
    require_224 = mode == "formal" and not args.config_only
    frame = load_prediction_frame(task_path, require_224=require_224)
    if args.task_limit is not None:
        if not args.allow_smoke or args.task_limit < 1:
            raise ValueError("task-limit is a positive smoke-only option")
        frame = frame.iloc[: args.task_limit].copy()
    checkpoint = choose_checkpoint(run_dir, status, args.checkpoint)
    seed = int(status["seed"])

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_frame_path = output_dir / "E216_PREDICTION_TASKS.csv"
    frame.to_csv(prediction_frame_path, index=False)
    cfg = build_prediction_config(
        training_config_path,
        checkpoint=checkpoint,
        task_frame_path=prediction_frame_path,
        output_dir=output_dir / "raw_predictions",
        control_asset=control_asset,
        seed=seed,
        max_control_cells=args.max_control_cells,
        chunk_size=args.chunk_size,
        accelerator=args.accelerator,
    )
    resolved_config_path = output_dir / "E216_PREDICTION_CONFIG.yaml"
    OmegaConf.save(cfg, resolved_config_path, resolve=True)
    prediction_status_path = output_dir / "E216_PREDICTION_STATUS.json"
    prediction_status = {
        "experiment": "E216_jiang24_resource_bounded_confirmation",
        "stage": "D2_TRUTH_FREE_PREDICTION",
        "status": "CONFIG_ONLY" if args.config_only else "RUNNING",
        "started_at": now_iso(),
        "architecture": status.get("architecture"),
        "seed": seed,
        "training_run": str(run_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "task_count": int(len(frame)),
        "task_inventory_sha256": sha256(task_path),
        "prediction_frame_sha256": sha256(prediction_frame_path),
        "control_asset": str(control_asset),
        "control_asset_sha256": sha256(control_asset),
        "resolved_prediction_config_sha256": sha256(resolved_config_path),
        "max_control_cells_per_state": 128,
        "prediction_control_seed": PREDICTION_CONTROL_SEED,
        "aggregation": "arithmetic_gene_mean_per_task",
        "test_truth_access": "NOT_AUTHORIZED",
        "test_perturbed_expression_rows_read": 0,
    }
    atomic_json(prediction_status_path, prediction_status)
    if args.config_only:
        return 0

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    perturbench_src = Path(args.perturbench_repo).resolve() / "src"
    sys.path.insert(0, str(perturbench_src))
    safeconf_src = Path(__file__).resolve().parents[2] / "code" / "safeconf_audit"
    sys.path.insert(0, str(safeconf_src))
    try:
        from perturbench.modelcore.predict import predict

        predict(cfg)
        outputs = sorted((output_dir / "raw_predictions").glob("prediction_rank*_chunk_*.h5ad"))
        if not outputs:
            raise RuntimeError("PerturBench produced no prediction chunks")
        centroid_path = output_dir / "E216_TASK_CENTROID_PREDICTIONS.npz"
        task_summary = aggregate_prediction_chunks(outputs, frame, centroid_path)
        task_summary_path = output_dir / "E216_TASK_PREDICTION_COUNTS.csv"
        task_summary.to_csv(task_summary_path, index=False)
        control_path = output_dir / "E216_CONTROL_CENTROIDS.npz"
        control_summary = build_control_centroids(cfg, frame, control_path)
        control_summary_path = output_dir / "E216_CONTROL_COUNTS.csv"
        control_summary.to_csv(control_summary_path, index=False)
        prediction_status.update(
            {
                "status": "COMPLETE",
                "finished_at": now_iso(),
                "chunk_count": len(outputs),
                "raw_prediction_bytes": sum(path.stat().st_size for path in outputs),
                "raw_prediction_sha256": {path.name: sha256(path) for path in outputs},
                "task_centroid_sha256": sha256(centroid_path),
                "task_prediction_counts_sha256": sha256(task_summary_path),
                "control_centroid_sha256": sha256(control_path),
                "control_counts_sha256": sha256(control_summary_path),
            }
        )
        atomic_json(prediction_status_path, prediction_status)
        return 0
    except Exception as exc:
        prediction_status.update(
            {
                "status": "FAILED",
                "finished_at": now_iso(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        atomic_json(prediction_status_path, prediction_status)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
