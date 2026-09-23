#!/usr/bin/env python3
"""Train one preregistered E233 corrected Jiang24 linear predictor."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

PERTURBENCH_COMMIT = "c84038bc1ea409aa54f3832cfa6f34f5059adf0c"
H5_BYTES = 93_532_364_449
SPLIT_SHA256 = "5af7da86a5b3994d570c0b1957d91f17cebb9f9b1943bac738b74fdb14b2ef5d"
VARIANTS = {"matched_control_softplus": True, "matched_control_linear": False}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perturbench-repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.seed != 1:
        raise RuntimeError("E233 stage 1 registers seed 1 only")
    repo = args.perturbench_repo.resolve()
    data = args.data_dir.resolve()
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    if subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip() != PERTURBENCH_COMMIT:
        raise RuntimeError("PerturBench commit changed")
    if subprocess.run(["git", "-C", str(repo), "diff", "--quiet"], check=False).returncode:
        raise RuntimeError("PerturBench tracked source is dirty")
    h5ad, split = data / "jiang24_processed.h5ad", data / "jiang24_split.csv"
    if h5ad.stat().st_size != H5_BYTES or sha256(split) != SPLIT_SHA256:
        raise RuntimeError("Jiang24 inputs changed")
    status = {
        "experiment": "E233_corrected_jiang24_predictor", "stage": "STAGE1_TRAINING",
        "status": "RUNNING", "started_at": datetime.now().astimezone().isoformat(),
        "variant": args.variant, "seed": args.seed, "test_perturbed_expression_rows_read": 0,
        "perturbench_commit": PERTURBENCH_COMMIT, "split_sha256": SPLIT_SHA256,
    }
    atomic_json(run / "E233_RUN_STATUS.json", status)
    command = [
        str(args.python.expanduser().absolute()), str(repo / "src/perturbench/modelcore/train.py"),
        "experiment=neurips2025/jiang24/linear_best_params_jiang24", "train=true", "test=false",
        f"seed={args.seed}", f"paths.data_dir={data}", f"paths.log_dir={run}",
        "data._target_=perturbench.data.modules.H5LitModule",
        "data.data_iter_factory._target_=perturbench.data.datasets.h5.SingleCellPerturbationWithControls.from_h5",
        "+data.data_iter_factory.cache_size=0", "+data.loader.persistent_workers=true",
        "data.loader.batch_size=2000", "data.loader.num_workers=8",
        "model._target_=e233_corrected_models.CorrectedLinearAdditive",
        f"+model.softplus_output={'true' if VARIANTS[args.variant] else 'false'}",
        "trainer.max_epochs=400", "trainer.min_epochs=5", "trainer.deterministic=true",
        "callbacks.early_stopping.patience=50", "+callbacks.model_checkpoint.save_last=true",
        f"hydra.run.dir={run}",
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parent), str(repo / "src")])
    environment["HDF5_USE_FILE_LOCKING"] = "FALSE"
    environment["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    try:
        with (run / "formal_train.log").open("a") as log:
            completed = subprocess.run(command, cwd=repo, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, check=False)
        if completed.returncode:
            raise RuntimeError(f"training exited with {completed.returncode}")
        config = yaml.safe_load((run / ".hydra/config.yaml").read_text())
        checks = {
            "test_disabled": config["test"] is False,
            "model": config["model"]["_target_"] == "e233_corrected_models.CorrectedLinearAdditive",
            "softplus": config["model"]["softplus_output"] is VARIANTS[args.variant],
            "dataset": config["data"]["data_iter_factory"]["_target_"].endswith("SingleCellPerturbationWithControls.from_h5"),
        }
        if not all(checks.values()):
            raise RuntimeError(f"resolved config failed: {checks}")
        checkpoints = sorted((run / "checkpoints").glob("*.ckpt"))
        if not checkpoints:
            raise RuntimeError("no checkpoints")
        status.update(status="COMPLETE", finished_at=datetime.now().astimezone().isoformat(),
                      config_sha256=sha256(run / ".hydra/config.yaml"),
                      checkpoints=[{"path": str(p), "sha256": sha256(p), "bytes": p.stat().st_size} for p in checkpoints])
    except BaseException as exc:
        status.update(status="FAILED", finished_at=datetime.now().astimezone().isoformat(), reason=repr(exc))
        raise
    finally:
        atomic_json(run / "E233_RUN_STATUS.json", status)


if __name__ == "__main__":
    main()
