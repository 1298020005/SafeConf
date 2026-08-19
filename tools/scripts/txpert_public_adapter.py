#!/usr/bin/env python3
"""Run frozen TxPert public checkpoints or its general baseline without auto-download.

The upstream ``main.py`` downloads every Zenodo asset on every entry through its
decorated CLI.  This adapter calls the upstream ``infer`` function directly, so
E199 only needs the two preregistered K562 archives.  It does not copy or patch
TxPert source code.  Raw predictions remain outside the SafeConf repository.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path


ALLOWED_CONFIGS = {
    "config-gat",
    "config-exphormer",
    "config-exphormer-mg",
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--txpert-repo", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument(
        "--kind",
        choices=("checkpoint", "general-baseline", "leakage-smoke"),
        required=True,
    )
    parser.add_argument("--config-name", default="config-gat")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def require_repo(repo: Path) -> None:
    required = (
        repo / "main.py",
        repo / "configs/config-gat.yaml",
        repo / "gspp/data/datamodule.py",
        repo / "gspp/models/baselines.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing TxPert inputs: " + ", ".join(missing))


def compose(repo: Path, config_name: str, overrides: list[str]):
    from hydra import compose as hydra_compose
    from hydra import initialize_config_dir

    with initialize_config_dir(
        config_dir=str((repo / "configs").resolve()), version_base="1.3"
    ):
        return hydra_compose(config_name=config_name, overrides=overrides)


def run_checkpoint(args: argparse.Namespace) -> dict:
    if args.config_name not in ALLOWED_CONFIGS:
        raise ValueError(
            f"checkpoint config must be one of {sorted(ALLOWED_CONFIGS)}"
        )
    cfg = compose(
        args.txpert_repo,
        args.config_name,
        [
            "mode=predict",
            f"save_dir={args.save_dir.resolve()}",
            f"datamodule.batch_size={args.batch_size}",
        ],
    )
    from main import infer

    infer(cfg)
    return {
        "kind": "checkpoint",
        "config_name": args.config_name,
        "checkpoint_name": str(cfg.checkpoint_name),
        "seed": int(cfg.seed),
        "batch_size": int(cfg.datamodule.batch_size),
        "task_type": str(cfg.datamodule.task_type),
    }


def run_leakage_smoke(args: argparse.Namespace) -> dict:
    """Prove on one real batch that replacing target expression changes no prediction."""
    import hashlib

    import numpy as np
    import torch
    from omegaconf import OmegaConf

    from gspp.data.datamodule import PertDataModule
    from gspp.data.graphmodule import GSPGraph
    from gspp.predictor import PertPredictor
    from gspp.utils import set_seed

    if args.config_name not in ALLOWED_CONFIGS:
        raise ValueError(
            f"leakage-smoke config must be one of {sorted(ALLOWED_CONFIGS)}"
        )
    cfg = compose(
        args.txpert_repo,
        args.config_name,
        [
            "mode=predict",
            f"datamodule.batch_size={min(args.batch_size, 16)}",
        ],
    )
    set_seed(int(cfg.seed))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    datamodule = PertDataModule(
        **OmegaConf.to_container(cfg.datamodule, resolve=True)
    )
    datamodule.prepare_data()
    datamodule.setup("test")
    graph = GSPGraph(
        pert2id=datamodule.pert2id,
        gene2id=datamodule.gene2id,
        **cfg.graph,
    )
    model = PertPredictor.load_from_checkpoint(
        args.txpert_repo / "cache/checkpoints" / str(cfg.checkpoint_name),
        input_dim=datamodule.input_dim,
        output_dim=datamodule.output_dim,
        adata_output_dim=datamodule.adata_output_dim,
        num_perts=len(datamodule.pert2id),
        graph=graph,
        pert_names=list(datamodule.pert2id.keys()),
        model_args=OmegaConf.to_container(cfg.model, resolve=True),
        device=device,
    ).to(device)
    model.eval()
    batch = next(iter(datamodule.predict_dataloader()))

    def array_hash(tensor: torch.Tensor) -> str:
        data = np.ascontiguousarray(tensor.detach().cpu().numpy()).tobytes()
        return hashlib.sha256(data).hexdigest()

    control_hash_before = array_hash(batch.control)
    truth_hash_before = array_hash(batch.x)
    with torch.no_grad():
        prediction_before = model.sample_inference(
            batch.control, batch.pert_idxs, batch.p, batch.cell_types
        )
        batch.x.zero_()
        prediction_after = model.sample_inference(
            batch.control, batch.pert_idxs, batch.p, batch.cell_types
        )
    max_abs_delta = float(
        torch.max(torch.abs(prediction_before - prediction_after)).detach().cpu()
    )
    exact_equal = bool(torch.equal(prediction_before, prediction_after))
    control_hash_after = array_hash(batch.control)
    if not exact_equal or max_abs_delta != 0.0:
        raise RuntimeError(
            "prediction changed after target expression was zeroed: "
            f"exact_equal={exact_equal}, max_abs_delta={max_abs_delta}"
        )
    if control_hash_before != control_hash_after:
        raise RuntimeError("control tensor changed during leakage smoke test")

    args.save_dir.mkdir(parents=True, exist_ok=False)
    return {
        "kind": "leakage-smoke",
        "config_name": args.config_name,
        "checkpoint_name": str(cfg.checkpoint_name),
        "seed": int(cfg.seed),
        "batch_size": int(batch.x.shape[0]),
        "n_genes": int(batch.x.shape[1]),
        "truth_hash_before_zeroing": truth_hash_before,
        "control_hash_before": control_hash_before,
        "control_hash_after": control_hash_after,
        "prediction_hash_before": array_hash(prediction_before),
        "prediction_hash_after": array_hash(prediction_after),
        "prediction_exact_equal": exact_equal,
        "prediction_max_abs_delta": max_abs_delta,
    }


def run_general_baseline(args: argparse.Namespace) -> dict:
    import anndata as ad
    import numpy as np
    import pandas as pd
    import torch
    from omegaconf import OmegaConf

    from gspp.data.datamodule import PertDataModule
    from gspp.models.baselines import MeanBaseline
    from gspp.utils import set_seed

    cfg = compose(
        args.txpert_repo,
        "config-baseline",
        [f"datamodule.batch_size={args.batch_size}"],
    )
    set_seed(int(cfg.seed))
    data_args = OmegaConf.to_container(cfg.datamodule, resolve=True)
    datamodule = PertDataModule(**data_args)
    datamodule.prepare_data()
    datamodule.setup("fit")

    baseline = MeanBaseline()
    baseline.prepare_baseline(**baseline.parse_inputs(datamodule))
    results = baseline.apply_baseline(datamodule.test_data)

    test_ds = datamodule.test_data
    controls = torch.stack(
        [test_ds._get_control_data(i) for i in range(len(test_ds))]
    ).detach().cpu().numpy()
    predictions = np.asarray(results["pred"], dtype=np.float32)
    truths = np.asarray(results["truth"], dtype=np.float32)
    if not (predictions.shape == truths.shape == controls.shape):
        raise RuntimeError(
            "baseline output/control/truth shapes differ: "
            f"{predictions.shape}, {truths.shape}, {controls.shape}"
        )

    condition_names = [
        f"{ct}_{pert}_{dose}"
        for ct, pert, dose in zip(
            results["cell_type"], results["pert_cat"], results["dose_cat"]
        )
    ]
    obs = pd.DataFrame(
        {
            "pert_cond_names": condition_names,
            "cell_types": results["cell_type"],
            "experimental_batches": results["experimental_batch"],
            "control": False,
        }
    )
    var = datamodule.adata.var[: datamodule.adata_output_dim].copy()
    args.save_dir.mkdir(parents=True, exist_ok=False)

    payload = {
        "base_state": controls,
        "output": predictions,
        "ground_truth": truths,
        "pert_cond_names": condition_names,
        "cell_types": list(results["cell_type"]),
        "experimental_batches": list(results["experimental_batch"]),
    }
    with (args.save_dir / "test_results.pkl").open("wb") as handle:
        pickle.dump(payload, handle)

    ad.AnnData(predictions, obs=obs, var=var).write_h5ad(
        args.save_dir / "test_predictions.h5ad"
    )
    ad.AnnData(truths, obs=obs, var=var).write_h5ad(
        args.save_dir / "test_ground_truth.h5ad"
    )
    control_obs = obs.copy()
    control_obs["control"] = True
    ad.AnnData(controls, obs=control_obs, var=var).write_h5ad(
        args.save_dir / "test_controls.h5ad"
    )
    return {
        "kind": "general-baseline",
        "config_name": "config-baseline",
        "seed": int(cfg.seed),
        "batch_size": int(cfg.datamodule.batch_size),
        "task_type": str(cfg.datamodule.task_type),
        "n_samples": int(predictions.shape[0]),
        "n_genes": int(predictions.shape[1]),
    }


def main() -> None:
    args = parse_args()
    args.txpert_repo = args.txpert_repo.resolve()
    args.save_dir = args.save_dir.resolve()
    require_repo(args.txpert_repo)
    if args.save_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.save_dir}")

    os.chdir(args.txpert_repo)
    sys.path.insert(0, str(args.txpert_repo))
    started = now()
    if args.kind == "checkpoint":
        metadata = run_checkpoint(args)
    elif args.kind == "general-baseline":
        metadata = run_general_baseline(args)
    else:
        metadata = run_leakage_smoke(args)
    metadata.update(
        {
            "started_at": started,
            "finished_at": now(),
            "txpert_repo": str(args.txpert_repo),
            "save_dir": str(args.save_dir),
        }
    )
    (args.save_dir / "E199_ADAPTER_RUN.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
