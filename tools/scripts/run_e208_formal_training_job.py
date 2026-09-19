#!/usr/bin/env python3
"""Run one preregistered, truth-isolated E208 Jiang24 training job.

This wrapper keeps PerturBench's published hyperparameters, replaces the
AnnData loader with its lazy H5 loader, disables testing, and writes a small
machine-readable completion record.  It never asks the data module for the
test stage.
"""
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
SPLIT_SHA256 = "5af7da86a5b3994d570c0b1957d91f17cebb9f9b1943bac738b74fdb14b2ef5d"
EXPERIMENTS = {
    "latent": "neurips2025/jiang24/latent_best_params_jiang24",
    "linear": "neurips2025/jiang24/linear_best_params_jiang24",
}
MODEL_TARGETS = {
    "latent": "perturbench.modelcore.models.LatentAdditive",
    "linear": "perturbench.modelcore.models.LinearAdditive",
}
REGISTERED_SEEDS = {"latent": {1, 2, 3, 4}, "linear": {1}}
REGISTERED_BATCH_SIZES = {2000, 1000, 500, 250}


class FormalJobFailure(RuntimeError):
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
    run_dir: Path,
    architecture: str,
    seed: int,
    batch_size: int,
    resume_checkpoint: Path | None = None,
) -> list[str]:
    if architecture not in EXPERIMENTS:
        raise FormalJobFailure(f"unregistered architecture: {architecture}")
    if seed not in REGISTERED_SEEDS[architecture]:
        raise FormalJobFailure(f"unregistered seed for {architecture}: {seed}")
    if batch_size not in REGISTERED_BATCH_SIZES:
        raise FormalJobFailure(f"unregistered batch size: {batch_size}")
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
        "+data.data_iter_factory.cache_size=0",
        "+data.loader.persistent_workers=true",
        f"data.loader.batch_size={batch_size}",
        "data.loader.num_workers=8",
        "trainer.max_epochs=400",
        "trainer.min_epochs=5",
        "trainer.deterministic=true",
        "callbacks.early_stopping.patience=50",
        "+callbacks.model_checkpoint.save_last=true",
        f"hydra.run.dir={run_dir}",
    ]
    if resume_checkpoint is not None:
        command.append(f"ckpt_path={resume_checkpoint}")
    return command


def validate_frozen_inputs(repo: Path, data_dir: Path) -> dict:
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != PERTURBENCH_COMMIT:
        raise FormalJobFailure(f"PerturBench commit changed: {commit}")
    if subprocess.run(
        ["git", "-C", str(repo), "diff", "--quiet"], check=False
    ).returncode:
        raise FormalJobFailure("PerturBench tracked source is dirty")
    h5ad = data_dir / "jiang24_processed.h5ad"
    split = data_dir / "jiang24_split.csv"
    if not h5ad.is_file() or h5ad.stat().st_size != H5_BYTES:
        raise FormalJobFailure("Jiang24 H5 size changed")
    if not split.is_file() or sha256_file(split) != SPLIT_SHA256:
        raise FormalJobFailure("Jiang24 split hash changed")
    return {
        "perturbench_commit": commit,
        "h5ad_bytes": h5ad.stat().st_size,
        "split_sha256": SPLIT_SHA256,
    }


def validate_resolved_config(
    path: Path, architecture: str, seed: int, batch_size: int
) -> None:
    if not path.is_file():
        raise FormalJobFailure(f"missing resolved Hydra config: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    data = config.get("data", {})
    iterator = data.get("data_iter_factory", {})
    trainer = config.get("trainer", {})
    callbacks = config.get("callbacks", {})
    checks = {
        "train_enabled": config.get("train") is True,
        "test_disabled": config.get("test") is False,
        "seed": config.get("seed") == seed,
        "model": config.get("model", {}).get("_target_")
        == MODEL_TARGETS[architecture],
        "h5_module": data.get("_target_")
        == "perturbench.data.modules.H5LitModule",
        "h5_iterator": iterator.get("_target_")
        == (
            "perturbench.data.datasets.h5."
            "SingleCellPerturbationWithControls.from_h5"
        ),
        "zero_cache": iterator.get("cache_size") == 0,
        "batch_size": data.get("loader", {}).get("batch_size") == batch_size,
        "workers": data.get("loader", {}).get("num_workers") == 8,
        "max_epochs": trainer.get("max_epochs") == 400,
        "min_epochs": trainer.get("min_epochs") == 5,
        "deterministic": trainer.get("deterministic") is True,
        "patience": callbacks.get("early_stopping", {}).get("patience") == 50,
        "save_last": callbacks.get("model_checkpoint", {}).get("save_last") is True,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise FormalJobFailure(f"resolved formal config changed: {failed}")


def reject_truth_or_test_outputs(run_dir: Path) -> None:
    forbidden = {"evaluation", "test", "truth", "target_truth"}
    bad = [
        path
        for path in run_dir.rglob("*")
        if forbidden.intersection(
            {part.lower() for part in path.relative_to(run_dir).parts}
        )
    ]
    if bad:
        raise FormalJobFailure(f"forbidden test/truth output created: {bad[0]}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perturbench-repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--architecture", choices=sorted(EXPERIMENTS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    repo = args.perturbench_repo.resolve()
    # Preserve the venv entry-point symlink; resolving it selects the base
    # interpreter and loses the registered PerturBench environment.
    python = args.python.expanduser().absolute()
    data_dir = args.data_dir.resolve()
    run_dir = args.run_dir.resolve()
    if not python.is_file():
        raise FormalJobFailure(f"missing Python: {python}")
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
        raise FormalJobFailure(f"PerturBench Python environment changed: {versions}")
    if args.architecture not in REGISTERED_SEEDS or args.seed not in REGISTERED_SEEDS[args.architecture]:
        raise FormalJobFailure("architecture/seed is outside the preregistration")
    lineage = validate_frozen_inputs(repo, data_dir)
    lineage.update(
        {
            "python_executable": str(python),
            "environment_versions": versions,
        }
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "E208_RUN_STATUS.json"
    status: dict[str, object] = {
        "experiment": "E208_jiang24_external_confirmation",
        "stage": "D1_FORMAL_TRAINING",
        "status": "RUNNING",
        "started_at": now(),
        "architecture": args.architecture,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "lineage": lineage,
        "test_truth_access": "NOT_AUTHORIZED",
        "test_perturbed_expression_rows_read": 0,
    }
    atomic_json(status_path, status)
    command = build_command(
        python=python,
        perturbench_repo=repo,
        data_dir=data_dir,
        run_dir=run_dir,
        architecture=args.architecture,
        seed=args.seed,
        batch_size=args.batch_size,
        resume_checkpoint=(
            args.resume_checkpoint.resolve() if args.resume_checkpoint else None
        ),
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
        with (run_dir / "formal_train.log").open("a", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=repo,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode:
            raise FormalJobFailure(f"training exited with code {completed.returncode}")
        config_path = run_dir / ".hydra/config.yaml"
        validate_resolved_config(
            config_path, args.architecture, args.seed, args.batch_size
        )
        reject_truth_or_test_outputs(run_dir)
        checkpoint_dir = run_dir / "checkpoints"
        last = checkpoint_dir / "last.ckpt"
        candidates = sorted(
            path
            for path in checkpoint_dir.glob("*.ckpt")
            if path.name != "last.ckpt" and path.stat().st_size > 0
        )
        if not last.is_file() or last.stat().st_size <= 0 or not candidates:
            raise FormalJobFailure("formal training did not preserve last and best checkpoints")
        status.update(
            {
                "status": "COMPLETE",
                "finished_at": now(),
                "resolved_config_sha256": sha256_file(config_path),
                "last_checkpoint": str(last),
                "last_checkpoint_bytes": last.stat().st_size,
                "best_checkpoint": str(candidates[-1]),
                "best_checkpoint_bytes": candidates[-1].stat().st_size,
                "test_truth_access": "NOT_AUTHORIZED",
                "test_perturbed_expression_rows_read": 0,
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
