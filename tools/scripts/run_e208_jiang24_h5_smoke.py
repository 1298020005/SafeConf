#!/usr/bin/env python3
"""Run the truth-isolated Jiang24 H5 training smoke after E205 completes.

The smoke uses PerturBench's own lazy H5 dataset with matched controls.  It
allows one train batch and one validation batch, forces ``test=false``, and
then starts a second process that restores the generated ``last.ckpt`` without
advancing beyond the one-epoch smoke limit.  Test-perturbation expression is
never requested by this runner.
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
H5_SHA256 = "5d876c0fa5770dc632ad8ed8b211ad7aef00ccc93481f6ac029d439a0d7cd4d9"
SPLIT_SHA256 = "5af7da86a5b3994d570c0b1957d91f17cebb9f9b1943bac738b74fdb14b2ef5d"
EXPERIMENTS = {
    "latent": "neurips2025/jiang24/latent_best_params_jiang24",
    "linear": "neurips2025/jiang24/linear_best_params_jiang24",
}


class SmokeFailure(RuntimeError):
    """A precondition, training, checkpoint, or no-truth gate failed."""


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


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
    batch_size: int,
    resume_checkpoint: Path | None = None,
) -> list[str]:
    """Build the frozen one-batch Hydra command without executing it."""
    if architecture not in EXPERIMENTS:
        raise SmokeFailure(f"unknown architecture: {architecture}")
    if batch_size not in {2000, 1000, 500, 250}:
        raise SmokeFailure("batch size is outside the preregistered fallback sequence")
    command = [
        str(python),
        str(perturbench_repo / "src/perturbench/modelcore/train.py"),
        f"experiment={EXPERIMENTS[architecture]}",
        "test=false",
        "train=true",
        "seed=1",
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
        "trainer.deterministic=true",
        "trainer.max_epochs=1",
        "trainer.min_epochs=1",
        "+trainer.limit_train_batches=1",
        "+trainer.limit_val_batches=1",
        "+trainer.num_sanity_val_steps=0",
        "+callbacks.model_checkpoint.save_last=true",
        f"hydra.run.dir={run_dir}",
    ]
    if resume_checkpoint is not None:
        command.append(f"ckpt_path={resume_checkpoint}")
    return command


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perturbench-repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--e205-queue-status", type=Path, required=True)
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--min-free-mb", type=int, default=22_000)
    parser.add_argument(
        "--allow-disjoint-running-e205",
        action="store_true",
        help=(
            "engineering-only preflight on a GPU not used by a still-blind E205 "
            "queue; this status must not release the formal E208 queue"
        ),
    )
    parser.add_argument(
        "--rehash-h5",
        action="store_true",
        help="recompute the 93 GB H5 SHA-256 in addition to the frozen D0 record",
    )
    return parser.parse_args(argv)


def check_e205_gate(
    path: Path,
    *,
    cuda_device: str,
    allow_disjoint_running: bool,
) -> tuple[dict, str]:
    if not path.is_file():
        raise SmokeFailure(f"missing E205 queue status: {path}")
    status = json.loads(path.read_text(encoding="utf-8"))
    common = (
        status.get("experiment") != "E205_cross_family_exphormer"
        or status.get("permanent_failures") not in ([], None)
        or status.get("target_truth_access") != "NOT_AUTHORIZED"
    )
    if common:
        raise SmokeFailure("E205 blind-state contract changed")
    if status.get("status") == "COMPLETE" and int(status.get("completed", -1)) == 16:
        return status, "AFTER_E205_COMPLETE"
    if not allow_disjoint_running:
        raise SmokeFailure("E205 has not completed cleanly in blind state")
    active_devices = {
        str(item.get("device"))
        for item in status.get("active", [])
        if isinstance(item, dict)
    }
    if (
        status.get("status") != "RUNNING"
        or int(status.get("completed", -1)) < 1
        or int(status.get("waiting", -1)) != 0
        or status.get("detached") not in ([], None)
        or str(cuda_device) in active_devices
    ):
        raise SmokeFailure("E205 does not permit a disjoint engineering preflight")
    return status, "DISJOINT_ENGINEERING_PREFLIGHT"


def check_inputs(args: argparse.Namespace) -> dict:
    repo = args.perturbench_repo.resolve()
    # ``venv/bin/python`` is a symlink.  ``resolve()`` would replace it with the
    # base interpreter and silently discard the virtual environment.
    python = args.python.expanduser().absolute()
    data = args.data_dir.resolve()
    if not python.is_file() or not (repo / "src/perturbench/modelcore/train.py").is_file():
        raise SmokeFailure("PerturBench Python or training entry point is missing")
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
        raise SmokeFailure(f"PerturBench Python environment changed: {versions}")
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != PERTURBENCH_COMMIT:
        raise SmokeFailure(f"PerturBench commit changed: {commit}")
    if subprocess.run(
        ["git", "-C", str(repo), "diff", "--quiet"], check=False
    ).returncode:
        raise SmokeFailure("PerturBench tracked source is dirty")

    h5ad = data / "jiang24_processed.h5ad"
    split = data / "jiang24_split.csv"
    d0 = data / "task_inventory/E208_TASK_INVENTORY_STATUS.json"
    if not h5ad.is_file() or h5ad.stat().st_size != H5_BYTES:
        raise SmokeFailure("Jiang24 H5 size changed")
    if not split.is_file() or sha256_file(split) != SPLIT_SHA256:
        raise SmokeFailure("Jiang24 split hash changed")
    if not d0.is_file():
        raise SmokeFailure("Jiang24 D0 inventory status is missing")
    inventory = json.loads(d0.read_text(encoding="utf-8"))
    if (
        inventory.get("status") != "PASS"
        or inventory.get("matrix_shape") != [1_628_476, 15_473]
        or inventory.get("h5ad_sha256") != H5_SHA256
        or inventory.get("split_sha256") != SPLIT_SHA256
        or inventory.get("expression_values_read") is not False
        or inventory.get("target_truth_used_for_selection") is not False
    ):
        raise SmokeFailure("Jiang24 D0 inventory contract changed")
    observed_h5_hash = sha256_file(h5ad) if args.rehash_h5 else H5_SHA256
    if observed_h5_hash != H5_SHA256:
        raise SmokeFailure("Jiang24 H5 hash changed")
    return {
        "perturbench_commit": commit,
        "python_executable": str(python),
        "environment_versions": versions,
        "h5ad_bytes": h5ad.stat().st_size,
        "h5ad_sha256": observed_h5_hash,
        "h5ad_rehashed_now": bool(args.rehash_h5),
        "split_sha256": SPLIT_SHA256,
        "inventory_sha256": sha256_file(d0),
    }


def free_gpu_mb(device: str) -> int:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={device}",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    return int(output.splitlines()[0].strip())


def validate_resolved_config(path: Path, architecture: str, batch_size: int) -> None:
    if not path.is_file():
        raise SmokeFailure(f"missing Hydra config: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected_model = {
        "latent": "perturbench.modelcore.models.LatentAdditive",
        "linear": "perturbench.modelcore.models.LinearAdditive",
    }[architecture]
    expected_iterator = (
        "perturbench.data.datasets.h5."
        "SingleCellPerturbationWithControls.from_h5"
    )
    checks = (
        config.get("train") is True,
        config.get("test") is False,
        config.get("seed") == 1,
        config.get("model", {}).get("_target_") == expected_model,
        config.get("data", {}).get("_target_")
        == "perturbench.data.modules.H5LitModule",
        config.get("data", {}).get("data_iter_factory", {}).get("_target_")
        == expected_iterator,
        config.get("data", {}).get("data_iter_factory", {}).get("cache_size") == 0,
        config.get("data", {}).get("loader", {}).get("batch_size") == batch_size,
        config.get("data", {}).get("loader", {}).get("num_workers") == 8,
        config.get("trainer", {}).get("max_epochs") == 1,
        config.get("trainer", {}).get("limit_train_batches") == 1,
        config.get("trainer", {}).get("limit_val_batches") == 1,
        config.get("trainer", {}).get("deterministic") is True,
    )
    if not all(checks):
        raise SmokeFailure(f"resolved {architecture} smoke config changed")


def reject_truth_or_test_outputs(run_dir: Path) -> None:
    forbidden = {"evaluation", "test", "truth", "target_truth"}
    bad = [
        path
        for path in run_dir.rglob("*")
        if forbidden.intersection({part.lower() for part in path.relative_to(run_dir).parts})
    ]
    if bad:
        raise SmokeFailure(f"forbidden test/truth output created: {bad[0]}")


def execute(command: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
    if completed.returncode:
        raise SmokeFailure(
            f"command failed with code {completed.returncode}; see {log_path}"
        )


def run_smoke(args: argparse.Namespace) -> dict:
    _, scheduling_gate = check_e205_gate(
        args.e205_queue_status.resolve(),
        cuda_device=str(args.cuda_device),
        allow_disjoint_running=bool(args.allow_disjoint_running_e205),
    )
    lineage = check_inputs(args)
    if args.output_root.exists():
        raise SmokeFailure(f"refusing to overwrite smoke output: {args.output_root}")
    free_before = free_gpu_mb(args.cuda_device)
    if free_before < args.min_free_mb:
        raise SmokeFailure(
            f"GPU {args.cuda_device} has {free_before} MiB free; need {args.min_free_mb}"
        )

    root = args.output_root.resolve()
    root.mkdir(parents=True)
    status: dict[str, object] = {
        "experiment": "E208_jiang24_external_confirmation",
        "stage": "D1_H5_ONE_BATCH_SMOKE",
        "status": "RUNNING",
        "started_at": now(),
        "batch_size": args.batch_size,
        "cuda_device": str(args.cuda_device),
        "free_gpu_mb_before": free_before,
        "lineage": lineage,
        "scheduling_gate": scheduling_gate,
        "formal_queue_release_authorized": scheduling_gate == "AFTER_E205_COMPLETE",
        "test_truth_access": "NOT_AUTHORIZED",
        "test_perturbed_expression_rows_read": 0,
        "architectures": {},
    }
    status_path = root / "E208_H5_SMOKE_STATUS.json"
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.cuda_device),
            "PYTHONPATH": str(args.perturbench_repo.resolve() / "src"),
            "HDF5_USE_FILE_LOCKING": "FALSE",
            "MLFLOW_DISABLE_AGENT_HINT": "1",
        }
    )

    try:
        for architecture in ("latent", "linear"):
            run_dir = root / architecture / "fit"
            command = build_command(
                python=args.python.expanduser().absolute(),
                perturbench_repo=args.perturbench_repo.resolve(),
                data_dir=args.data_dir.resolve(),
                run_dir=run_dir,
                architecture=architecture,
                batch_size=args.batch_size,
            )
            execute(command, root / architecture / "fit.log", env)
            validate_resolved_config(run_dir / ".hydra/config.yaml", architecture, args.batch_size)
            checkpoint = run_dir / "checkpoints/last.ckpt"
            if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
                raise SmokeFailure(f"{architecture} did not produce last.ckpt")
            reject_truth_or_test_outputs(run_dir)

            resume_dir = root / architecture / "resume"
            resume_command = build_command(
                python=args.python.expanduser().absolute(),
                perturbench_repo=args.perturbench_repo.resolve(),
                data_dir=args.data_dir.resolve(),
                run_dir=resume_dir,
                architecture=architecture,
                batch_size=args.batch_size,
                resume_checkpoint=checkpoint,
            )
            execute(resume_command, root / architecture / "resume.log", env)
            validate_resolved_config(
                resume_dir / ".hydra/config.yaml", architecture, args.batch_size
            )
            reject_truth_or_test_outputs(resume_dir)
            status["architectures"][architecture] = {
                "status": "PASS",
                "checkpoint_bytes": checkpoint.stat().st_size,
                "checkpoint_sha256": sha256_file(checkpoint),
                "fit_config_sha256": sha256_file(run_dir / ".hydra/config.yaml"),
                "resume_config_sha256": sha256_file(
                    resume_dir / ".hydra/config.yaml"
                ),
            }
        status["status"] = "PASS"
    except Exception as exc:
        status["status"] = "FAIL"
        status["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        status["finished_at"] = now()
        status["free_gpu_mb_after"] = free_gpu_mb(args.cuda_device)
        status_path.write_text(
            json.dumps(status, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return status


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_smoke(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
