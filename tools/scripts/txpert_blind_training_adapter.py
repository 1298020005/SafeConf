#!/usr/bin/env python3
"""Train public TxPert-GAT without making target perturbation truth accessible."""

from __future__ import annotations

import argparse
import hashlib
import json
import numbers
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path


PUBLIC_CELL_TYPES = ("K562", "RPE1", "hepg2", "jurkat")


class TrainingFailure(RuntimeError):
    pass


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--txpert-repo", type=Path, required=True)
    parser.add_argument("--task-type", required=True)
    parser.add_argument("--target", choices=PUBLIC_CELL_TYPES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--kind", choices=("smoke", "profile", "formal"), required=True
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--smoke-train-batches", type=int, default=20)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def git_text(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    repo = args.txpert_repo.resolve()
    run_dir = args.run_dir.resolve()
    adapter_path = Path(__file__).resolve()
    safeconf_repo = adapter_path.parents[2]
    if run_dir.exists():
        raise TrainingFailure(f"refusing to overwrite: {run_dir}")
    if git_text(repo, "rev-parse", "HEAD") != (
        "08d82eea86746b044cf7531f4ec8c5f60e1cb73f"
    ):
        raise TrainingFailure("unexpected TxPert commit")
    if git_text(repo, "status", "--porcelain"):
        raise TrainingFailure("TxPert source worktree is dirty")
    safeconf_head = git_text(safeconf_repo, "rev-parse", "HEAD")
    safeconf_branch = git_text(
        safeconf_repo, "rev-parse", "--abbrev-ref", "HEAD"
    )
    adapter_relpath = adapter_path.relative_to(safeconf_repo).as_posix()
    if git_text(
        safeconf_repo, "status", "--porcelain", "--", adapter_relpath
    ):
        raise TrainingFailure("SafeConf training adapter is not committed")
    tracking_commits = {
        remote: git_text(
            safeconf_repo, "rev-parse", f"{remote}/{safeconf_branch}"
        )
        for remote in ("origin", "github")
    }
    if any(commit != safeconf_head for commit in tracking_commits.values()):
        raise TrainingFailure("SafeConf GitHub/Gitee tracking refs do not match HEAD")
    cache = repo / "cache" / args.task_type
    view_manifest_path = cache / "E201_BLIND_VIEW_MANIFEST.json"
    if not view_manifest_path.is_file():
        raise TrainingFailure(f"missing blind view manifest: {view_manifest_path}")
    view_manifest = json.loads(view_manifest_path.read_text(encoding="utf-8"))
    if view_manifest.get("target") != args.target:
        raise TrainingFailure("blind view target mismatch")
    if int(view_manifest.get("n_target_treatments", -1)) != 0:
        raise TrainingFailure("blind view contains target perturbations")
    if view_manifest.get("uns_keys") != []:
        raise TrainingFailure("blind view contains unapproved uns metadata")

    run_dir.mkdir(parents=True)
    metadata = {
        "experiment": "E201_txpert_multitarget_retraining",
        "kind": args.kind,
        "status": "RUNNING",
        "started_at": now(),
        "target": args.target,
        "source_contexts": sorted(set(PUBLIC_CELL_TYPES) - {args.target}),
        "seed": args.seed,
        "task_type": args.task_type,
        "batch_size": args.batch_size,
        "txpert_commit": git_text(repo, "rev-parse", "HEAD"),
        "txpert_clean": True,
        "safeconf_commit": safeconf_head,
        "safeconf_branch": safeconf_branch,
        "safeconf_tracking_commits": tracking_commits,
        "training_adapter_path": adapter_relpath,
        "training_adapter_sha256": sha256_file(adapter_path),
        "training_adapter_committed": True,
        "blind_view_manifest": view_manifest,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "command": sys.argv,
    }
    write_json(run_dir / "E201_RUN_STATUS.json", metadata)

    os.chdir(repo)
    sys.path.insert(0, str(repo))
    try:
        import torch
        import pandas as pd
        from hydra import compose as hydra_compose
        from hydra import initialize_config_dir
        from lightning import Trainer
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
        from lightning.pytorch.loggers import CSVLogger
        from omegaconf import OmegaConf

        import gspp.constants as cs
        from gspp.data.datamodule import PertDataModule, XcelltypePerturbDataset
        from gspp.data.graphmodule import GSPGraph
        from gspp.predictor import PertPredictor
        from gspp.utils import set_seed

        class BlindXcelltypePerturbDataset(XcelltypePerturbDataset):
            def _extend_with_train_data(self, train_cond_names):
                n_controls = self.control_data_raw_X.shape[0]
                self.pert_data = torch.cat(
                    [self.pert_data, self.control_data_raw_X]
                )
                self.pert_conditions = pd.concat(
                    [
                        self.pert_conditions,
                        pd.Series([[-1]] * n_controls),
                    ]
                )
                self.condition_names = pd.concat(
                    [self.condition_names, train_cond_names]
                )
                self.index = pd.concat([self.index, self.control_index])
                self.treatment_cell_types = self.all_cell_types
                self.treatment_cell_batches = self.all_cell_batches

        class BlindPertDataModule(PertDataModule):
            def _create_datasets(self) -> None:
                target_mask = (
                    (self.treatment_data.obs[cs.CELL_TYPE] == self.test_cell_type)
                    & ~self.treatment_data.obs[cs.CONTROL].astype(bool)
                )
                if int(target_mask.sum()) != 0:
                    raise TrainingFailure("target treatment reached training datamodule")
                self.train_cell_type_mask = self.treatment_data.obs[
                    cs.CELL_TYPE
                ].isin(self.train_cell_types)
                train_indices = (
                    self.treatment_data.obs[cs.CONDITION]
                    .isin(self.condition_group["train"])
                    .to_numpy(bool)
                    & self.train_cell_type_mask.to_numpy(bool)
                )
                val_indices = (
                    self.treatment_data.obs[cs.CONDITION]
                    .isin(self.condition_group["val"])
                    .to_numpy(bool)
                    & self.train_cell_type_mask.to_numpy(bool)
                )
                if not train_indices.any() or not val_indices.any():
                    raise TrainingFailure("empty blind train or validation split")
                self.train_adata = self.treatment_data[train_indices]
                self.train_conditions = self.conditions[train_indices]
                self.train_condition_names = self.condition_names[train_indices]
                self.train_data = BlindXcelltypePerturbDataset(
                    self.train_adata,
                    self.control_data_train,
                    self.train_conditions,
                    self.train_condition_names,
                    self.device,
                    train_cond_names=self.control_train_condition_names,
                    match_cntr=self.match_cntr,
                    avg_cntr=self.avg_cntr,
                    obsm_key=self.obsm_key,
                )
                self.val_adata = self.treatment_data[val_indices]
                self.val_conditions = self.conditions[val_indices]
                self.val_condition_names = self.condition_names[val_indices]
                self.val_data = XcelltypePerturbDataset(
                    self.val_adata,
                    self.control_data_val,
                    self.val_conditions,
                    self.val_condition_names,
                    self.device,
                    match_cntr=self.match_cntr,
                    avg_cntr=self.avg_cntr,
                    obsm_key=self.obsm_key,
                )

        class BlindPertPredictor(PertPredictor):
            def __init__(self, *model_args, skip_validation_metric=False, **model_kwargs):
                super().__init__(*model_args, **model_kwargs)
                self.skip_validation_metric = skip_validation_metric

            def on_validation_epoch_end(self):
                if not self.skip_validation_metric:
                    self.general_validation_epoch(
                        cs.VAL, self.trainer.datamodule.val_dataloader()
                    )

        source_contexts = sorted(set(PUBLIC_CELL_TYPES) - {args.target})
        overrides = [
            "mode=baseline",
            f"datamodule.task_type={args.task_type}",
            "datamodule.train_cell_types=[" + ",".join(source_contexts) + "]",
            f"datamodule.test_cell_type={args.target}",
            "datamodule.val_cell_type=null",
            f"datamodule.batch_size={args.batch_size}",
            f"seed={args.seed}",
        ]
        with initialize_config_dir(
            config_dir=str((repo / "configs").resolve()), version_base="1.3"
        ):
            cfg = hydra_compose(
                config_name="config-x-cell-gat", overrides=overrides
            )
        set_seed(args.seed)
        if not torch.cuda.is_available():
            raise TrainingFailure("CUDA is required for E201 training")
        datamodule = BlindPertDataModule(
            **OmegaConf.to_container(cfg.datamodule, resolve=True)
        )
        datamodule.prepare_data()
        if set(datamodule.adata.uns.keys()) != {"pharos"}:
            unexpected = sorted(datamodule.adata.uns.keys())
            raise TrainingFailure(f"unexpected blind-view uns metadata: {unexpected}")
        datamodule.setup("fit")
        bad_index_rows = []
        n_control_targets = 0
        for row_index, perts in enumerate(datamodule.train_data.pert_conditions):
            valid = isinstance(perts, (list, tuple)) and all(
                isinstance(pert, numbers.Integral) for pert in perts
            )
            if not valid:
                bad_index_rows.append(
                    {"row": row_index, "value": repr(perts), "type": type(perts).__name__}
                )
                if len(bad_index_rows) >= 10:
                    break
            if valid and list(perts) == [-1]:
                n_control_targets += 1
        if bad_index_rows:
            raise TrainingFailure(f"non-integer perturbation indices: {bad_index_rows}")
        if n_control_targets != len(datamodule.control_data_train):
            raise TrainingFailure(
                "control target encoding count does not match allowed controls"
            )
        metadata.update(
            {
                "perturbation_index_contract": "all list/tuple of integral IDs",
                "control_target_id": -1,
                "n_control_training_targets": n_control_targets,
            }
        )
        graph = GSPGraph(
            pert2id=datamodule.pert2id,
            gene2id=datamodule.gene2id,
            **cfg.graph,
        )
        total_epochs = 80 if args.kind == "formal" else 1
        scheduler_args = (
            {"warmup_epochs": 0, "type": None, "total_epochs": 1}
            if args.kind != "formal"
            else {"warmup_epochs": 5, "type": "cosine", "total_epochs": 80}
        )
        model = BlindPertPredictor(
            input_dim=datamodule.input_dim,
            output_dim=datamodule.output_dim,
            adata_output_dim=datamodule.adata_output_dim,
            graph=graph,
            model_args=OmegaConf.to_container(cfg.model, resolve=True),
            lr=3e-4,
            min_lr=0.0,
            lr_scheduler_args=scheduler_args,
            weight_decay=0.0,
            device="cuda",
            run_val_on_train_data=False,
            match_cntr_for_eval=True,
            skip_validation_metric=args.kind == "smoke",
        ).to("cuda")
        checkpoint_dir = run_dir / "checkpoints"
        if args.kind == "formal":
            checkpoint = ModelCheckpoint(
                dirpath=checkpoint_dir,
                monitor="val_pearson_delta",
                mode="max",
                save_top_k=1,
                save_last=True,
                every_n_epochs=5,
                filename="best-{epoch:02d}-{val_pearson_delta:.5f}",
            )
            callbacks = [
                checkpoint,
                EarlyStopping(
                    monitor="val_pearson_delta", mode="max", patience=100
                ),
            ]
        else:
            checkpoint = ModelCheckpoint(
                dirpath=checkpoint_dir, save_top_k=0, save_last=True
            )
            callbacks = [checkpoint]
        logger = CSVLogger(save_dir=run_dir / "logs", name="txpert")
        trainer = Trainer(
            accelerator="gpu",
            devices=1,
            precision="32-true",
            max_epochs=total_epochs,
            callbacks=callbacks,
            logger=logger,
            deterministic=False,
            benchmark=False,
            num_sanity_val_steps=0,
            limit_train_batches=(
                args.smoke_train_batches if args.kind == "smoke" else 1.0
            ),
            limit_val_batches=0 if args.kind == "smoke" else 1.0,
            log_every_n_steps=20,
            enable_progress_bar=True,
        )
        torch.cuda.reset_peak_memory_stats()
        fit_started = time.perf_counter()
        trainer.fit(model, datamodule=datamodule)
        fit_wall_seconds = time.perf_counter() - fit_started
        checkpoint_files = []
        checkpoint_paths = {
            "last": Path(checkpoint.last_model_path)
            if checkpoint.last_model_path
            else None,
            "best_source_validation": Path(checkpoint.best_model_path)
            if checkpoint.best_model_path
            else None,
        }
        for role, path in checkpoint_paths.items():
            if path is None:
                continue
            if not path.is_file():
                raise TrainingFailure(f"missing {role} checkpoint: {path}")
            checkpoint_files.append(
                {
                    "role": role,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        metadata.update(
            {
                "status": "COMPLETE",
                "finished_at": now(),
                "torch_version": torch.__version__,
                "lightning_version": __import__("lightning").__version__,
                "n_train_dataset_rows": len(datamodule.train_data),
                "n_validation_dataset_rows": len(datamodule.val_data),
                "n_train_batches": len(datamodule.train_dataloader()),
                "n_validation_batches": len(datamodule.val_dataloader()),
                "max_epochs": total_epochs,
                "global_step": int(trainer.global_step),
                "current_epoch": int(trainer.current_epoch),
                "fit_wall_seconds": fit_wall_seconds,
                "cuda_peak_memory_allocated_bytes": int(
                    torch.cuda.max_memory_allocated()
                ),
                "cuda_peak_memory_reserved_bytes": int(
                    torch.cuda.max_memory_reserved()
                ),
                "best_model_path": checkpoint.best_model_path,
                "best_model_score": (
                    None
                    if checkpoint.best_model_score is None
                    else float(checkpoint.best_model_score.cpu())
                ),
                "last_model_path": checkpoint.last_model_path,
                "checkpoint_files": checkpoint_files,
                "target_test_dataset_constructed": False,
                "target_perturbed_cells_accessed": 0,
            }
        )
        write_json(run_dir / "E201_RUN_STATUS.json", metadata)
    except Exception as exc:
        metadata.update(
            {
                "status": "FAILED",
                "finished_at": now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        write_json(run_dir / "E201_RUN_STATUS.json", metadata)
        raise


if __name__ == "__main__":
    main()
