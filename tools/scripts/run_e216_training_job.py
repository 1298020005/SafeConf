#!/usr/bin/env python3
"""Run one truth-isolated resource-bounded Jiang24 training job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Sequence

import yaml


PERTURBENCH_COMMIT = "c84038bc1ea409aa54f3832cfa6f34f5059adf0c"
LIGHTNING_VERSION = "2.6.6"
TORCH_VERSION = "2.6.0+cu124"
H5_BYTES = 93_532_364_449
RESOURCE_SPLIT_SHA256 = "aa1261993e3456fe19db931cb9f20bd66febd4745bcd92144162c6d27afcf371"
HVG_SHA256 = "daec79a6c8584fbee0104f522749ac43b900f697a26696ee59988ef3d0aea7ee"
EXPERIMENTS = {
    "latent": "neurips2025/jiang24/latent_best_params_jiang24",
    "linear": "neurips2025/jiang24/linear_best_params_jiang24",
}
MODEL_TARGETS = {
    "latent": "perturbench.modelcore.models.LatentAdditive",
    "linear": "perturbench.modelcore.models.LinearAdditive",
}
REGISTERED_SEEDS = {"latent": {1, 2, 3, 4}, "linear": {1}}


class TrainingFailure(RuntimeError):
    """A frozen input, configuration, training, or checkpoint gate failed."""


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_command(
    *,
    python: Path,
    perturbench_repo: Path,
    data_dir: Path,
    assets_dir: Path,
    run_dir: Path,
    architecture: str,
    seed: int,
    mode: str,
    resume_checkpoint: Path | None = None,
) -> list[str]:
    if architecture not in REGISTERED_SEEDS or seed not in REGISTERED_SEEDS[architecture]:
        raise TrainingFailure("architecture/seed is outside the E216 registration")
    if mode not in {"smoke", "formal"}:
        raise TrainingFailure(f"unknown mode: {mode}")
    command = [
        str(python),
        str(perturbench_repo / "src/perturbench/modelcore/train.py"),
        f"experiment={EXPERIMENTS[architecture]}",
        "train=true",
        "test=false",
        f"seed={seed}",
        f"paths.data_dir={data_dir}",
        f"paths.log_dir={run_dir}",
        "data._target_=perturbench.data.modules.H5LitModule",
        (
            "data.data_iter_factory._target_="
            "perturbench.data.datasets.h5."
            "SingleCellPerturbationWithControls.from_h5"
        ),
        f"data.data_iter_factory.feature_filter_path={assets_dir / 'jiang24_official_hvg4000.csv'}",
        f"data.splitter.split_path={assets_dir / 'jiang24_e216_resource_split.csv'}",
        "+data.data_iter_factory.cache_size=0",
        "+data.loader.persistent_workers=true",
        "data.loader.batch_size=2000",
        "data.loader.num_workers=8",
        "trainer.deterministic=true",
        "+callbacks.model_checkpoint.save_last=true",
        f"hydra.run.dir={run_dir}",
    ]
    if mode == "smoke":
        command.extend(
            [
                "trainer.max_epochs=1",
                "trainer.min_epochs=1",
                "+trainer.limit_train_batches=1",
                "+trainer.limit_val_batches=1",
                "+trainer.num_sanity_val_steps=0",
                "callbacks.early_stopping.patience=1",
            ]
        )
    else:
        command.extend(
            [
                "trainer.max_epochs=10",
                "trainer.min_epochs=5",
                "callbacks.early_stopping.patience=3",
            ]
        )
    if resume_checkpoint is not None:
        command.append(f"ckpt_path={resume_checkpoint}")
    return command


def validate_inputs(repo: Path, data_dir: Path, assets_dir: Path) -> dict:
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != PERTURBENCH_COMMIT:
        raise TrainingFailure(f"PerturBench commit changed: {commit}")
    if subprocess.run(["git", "-C", str(repo), "diff", "--quiet"], check=False).returncode:
        raise TrainingFailure("PerturBench tracked source is dirty")
    h5ad = data_dir / "jiang24_processed.h5ad"
    split = assets_dir / "jiang24_e216_resource_split.csv"
    hvg = assets_dir / "jiang24_official_hvg4000.csv"
    status_path = assets_dir / "E216_RESOURCE_ASSET_STATUS.json"
    if not h5ad.is_file() or h5ad.stat().st_size != H5_BYTES:
        raise TrainingFailure("Jiang24 H5 identity changed")
    if sha256_file(split) != RESOURCE_SPLIT_SHA256 or sha256_file(hvg) != HVG_SHA256:
        raise TrainingFailure("E216 resource assets changed")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if (
        status.get("status") != "PASS"
        or status.get("expression_values_read") is not False
        or status.get("test_membership_unchanged") is not True
        or int(status.get("n_features", -1)) != 4000
    ):
        raise TrainingFailure("E216 resource asset gate failed")
    return {
        "perturbench_commit": commit,
        "h5ad_bytes": h5ad.stat().st_size,
        "resource_split_sha256": RESOURCE_SPLIT_SHA256,
        "hvg_sha256": HVG_SHA256,
    }


def validate_config(path: Path, architecture: str, seed: int, mode: str) -> None:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    data = config.get("data", {})
    iterator = data.get("data_iter_factory", {})
    trainer = config.get("trainer", {})
    callbacks = config.get("callbacks", {})
    checks = {
        "train_enabled": config.get("train") is True,
        "test_disabled": config.get("test") is False,
        "seed": config.get("seed") == seed,
        "model": config.get("model", {}).get("_target_") == MODEL_TARGETS[architecture],
        "h5_module": data.get("_target_") == "perturbench.data.modules.H5LitModule",
        "resource_split": str(data.get("splitter", {}).get("split_path", "")).endswith(
            "jiang24_e216_resource_split.csv"
        ),
        "hvg4000": str(iterator.get("feature_filter_path", "")).endswith(
            "jiang24_official_hvg4000.csv"
        ),
        "zero_cache": iterator.get("cache_size") == 0,
        "batch_size": data.get("loader", {}).get("batch_size") == 2000,
        "workers": data.get("loader", {}).get("num_workers") == 8,
        "deterministic": trainer.get("deterministic") is True,
        "save_last": callbacks.get("model_checkpoint", {}).get("save_last") is True,
    }
    if mode == "formal":
        checks.update(
            {
                "max_epochs": trainer.get("max_epochs") == 10,
                "min_epochs": trainer.get("min_epochs") == 5,
                "patience": callbacks.get("early_stopping", {}).get("patience") == 3,
            }
        )
    else:
        checks.update(
            {
                "max_epochs": trainer.get("max_epochs") == 1,
                "limit_train": trainer.get("limit_train_batches") == 1,
                "limit_val": trainer.get("limit_val_batches") == 1,
            }
        )
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise TrainingFailure(f"resolved E216 config changed: {failed}")


def reject_truth_outputs(run_dir: Path) -> None:
    forbidden = {"evaluation", "test", "truth", "target_truth"}
    for path in run_dir.rglob("*"):
        if forbidden.intersection(
            part.lower() for part in path.relative_to(run_dir).parts
        ):
            raise TrainingFailure(f"forbidden truth/test output created: {path}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perturbench-repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--architecture", choices=sorted(EXPERIMENTS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    repo = args.perturbench_repo.resolve()
    python = args.python.expanduser().absolute()
    data_dir = args.data_dir.resolve()
    assets_dir = args.assets_dir.resolve()
    run_dir = args.run_dir.resolve()
    versions = json.loads(
        subprocess.check_output(
            [
                str(python),
                "-c",
                (
                    "import json, lightning, torch; "
                    "print(json.dumps({'lightning': lightning.__version__, "
                    "'torch': torch.__version__}))"
                ),
            ],
            text=True,
        )
    )
    if versions != {"lightning": LIGHTNING_VERSION, "torch": TORCH_VERSION}:
        raise TrainingFailure(f"PerturBench environment changed: {versions}")
    lineage = validate_inputs(repo, data_dir, assets_dir)
    lineage.update({"python_executable": str(python), "environment_versions": versions})
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "E216_RUN_STATUS.json"
    status: dict[str, object] = {
        "experiment": "E216_jiang24_resource_bounded_confirmation",
        "stage": "D1_SMOKE" if args.mode == "smoke" else "D1_FORMAL_TRAINING",
        "status": "RUNNING",
        "started_at": now(),
        "architecture": args.architecture,
        "seed": args.seed,
        "mode": args.mode,
        "lineage": lineage,
        "test_truth_access": "NOT_AUTHORIZED",
        "test_perturbed_expression_rows_read": 0,
    }
    atomic_json(status_path, status)
    command = build_command(
        python=python,
        perturbench_repo=repo,
        data_dir=data_dir,
        assets_dir=assets_dir,
        run_dir=run_dir,
        architecture=args.architecture,
        seed=args.seed,
        mode=args.mode,
        resume_checkpoint=args.resume_checkpoint.resolve() if args.resume_checkpoint else None,
    )
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(repo / "src"),
            "HDF5_USE_FILE_LOCKING": "FALSE",
            "MLFLOW_DISABLE_AGENT_HINT": "1",
        }
    )
    try:
        with (run_dir / "train.log").open("a", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=repo,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode:
            raise TrainingFailure(f"training exited with code {completed.returncode}")
        config_path = run_dir / ".hydra/config.yaml"
        validate_config(config_path, args.architecture, args.seed, args.mode)
        reject_truth_outputs(run_dir)
        checkpoint_dir = run_dir / "checkpoints"
        last = checkpoint_dir / "last.ckpt"
        candidates = sorted(
            path
            for path in checkpoint_dir.glob("*.ckpt")
            if path.name != "last.ckpt" and path.stat().st_size > 0
        )
        if not last.is_file() or not candidates:
            raise TrainingFailure("last or best checkpoint is missing")
        status.update(
            {
                "status": "PASS" if args.mode == "smoke" else "COMPLETE",
                "finished_at": now(),
                "resolved_config_sha256": sha256_file(config_path),
                "last_checkpoint": str(last),
                "best_checkpoint": str(candidates[-1]),
            }
        )
    except Exception as exc:
        status.update(
            {
                "status": "FAILED",
                "failed_at": now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        atomic_json(status_path, status)
        raise
    atomic_json(status_path, status)
    return status


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
