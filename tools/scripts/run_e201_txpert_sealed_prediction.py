#!/usr/bin/env python3
"""Predict one E201 target/seed only after the 16-checkpoint family is sealed."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path


TARGETS = ("K562", "RPE1", "hepg2", "jurkat")
SEEDS = (1, 2, 3, 4)
TXPERT_COMMIT = "08d82eea86746b044cf7531f4ec8c5f60e1cb73f"
OFFICIAL_SOURCE_H5AD_SHA256 = (
    "1b557390148eba358304e43e0b239538d9ae0691b26ec843f41cf544960307a8"
)
BLIND_H5AD_SHA256 = "85f93d1b29ded34d9dcece9ecdba1ef722a3f14aeedbfbe740eed9f045fbe486"
BLIND_MANIFEST_SHA256 = (
    "27448df0378aab32e1a9fd22bf20c18c90089816cee6c28b9710cd2d6f812e7d"
)


class PredictionFailure(RuntimeError):
    pass


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--txpert-repo", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--family-seal", type=Path, required=True)
    parser.add_argument("--target", choices=TARGETS, required=True)
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    parser.add_argument(
        "--checkpoint-role",
        choices=("last", "best_source_validation"),
        default="last",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shared-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def git_text(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def resolve_data_path(value: str, data_root: Path) -> Path:
    prefix = "DATA/"
    if not value.startswith(prefix):
        raise PredictionFailure(f"family seal path is not DATA-relative: {value}")
    resolved = (data_root / value[len(prefix) :]).resolve()
    try:
        resolved.relative_to(data_root.resolve())
    except ValueError as exc:
        raise PredictionFailure(f"family seal path escapes data root: {value}") from exc
    return resolved


def compose(repo: Path, overrides: list[str]):
    from hydra import compose as hydra_compose
    from hydra import initialize_config_dir

    with initialize_config_dir(
        config_dir=str((repo / "configs").resolve()), version_base="1.3"
    ):
        return hydra_compose(config_name="config-x-cell-gat", overrides=overrides)


def array_bytes(value) -> bytes:
    import numpy as np

    return np.ascontiguousarray(value).tobytes()


def verify_shared_files(shared_dir: Path, manifest: dict) -> dict[str, Path]:
    entries = manifest.get("files", [])
    by_role = {entry.get("role"): entry for entry in entries}
    if set(by_role) != {"controls", "observations"}:
        raise PredictionFailure("shared target manifest has unexpected file roles")
    verified = {}
    expected_names = {
        "controls": "controls.npy",
        "observations": "observations.csv",
    }
    for role, expected_name in expected_names.items():
        entry = by_role[role]
        if entry.get("path") != expected_name:
            raise PredictionFailure(f"unexpected shared {role} path")
        path = (shared_dir / expected_name).resolve()
        if path.parent != shared_dir.resolve() or not path.is_file():
            raise PredictionFailure(f"missing shared {role} file")
        if path.stat().st_size != int(entry.get("bytes", -1)) or sha256_file(
            path
        ) != entry.get("sha256"):
            raise PredictionFailure(f"shared {role} file changed after sealing")
        verified[role] = path
    return verified


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise PredictionFailure("batch-size must be positive")
    txpert_repo = args.txpert_repo.resolve()
    data_root = args.data_root.resolve()
    family_seal_path = args.family_seal.resolve()
    output_dir = args.output_dir.resolve()
    shared_dir = args.shared_dir.resolve()
    script_path = Path(__file__).resolve()
    safeconf_repo = script_path.parents[2]

    if output_dir.exists():
        raise PredictionFailure(f"refusing to overwrite: {output_dir}")
    if git_text(txpert_repo, "rev-parse", "HEAD") != TXPERT_COMMIT:
        raise PredictionFailure("unexpected TxPert commit")
    if git_text(txpert_repo, "status", "--porcelain"):
        raise PredictionFailure("TxPert source worktree is dirty")
    if not family_seal_path.is_file():
        raise PredictionFailure(f"missing family seal: {family_seal_path}")
    try:
        seal_relpath = family_seal_path.relative_to(safeconf_repo).as_posix()
    except ValueError as exc:
        raise PredictionFailure(
            "family seal must be inside SafeConf repository"
        ) from exc
    script_relpath = script_path.relative_to(safeconf_repo).as_posix()
    for relpath in (seal_relpath, script_relpath):
        if git_text(safeconf_repo, "status", "--porcelain", "--", relpath):
            raise PredictionFailure(f"uncommitted release input: {relpath}")
    safeconf_head = git_text(safeconf_repo, "rev-parse", "HEAD")
    safeconf_branch = git_text(safeconf_repo, "rev-parse", "--abbrev-ref", "HEAD")
    tracking = {
        remote: git_text(safeconf_repo, "rev-parse", f"{remote}/{safeconf_branch}")
        for remote in ("origin", "github")
    }
    if any(value != safeconf_head for value in tracking.values()):
        raise PredictionFailure("SafeConf GitHub/Gitee tracking refs do not match HEAD")

    seal = json.loads(family_seal_path.read_text(encoding="utf-8"))
    if (
        seal.get("status") != "SEALED_16_CHECKPOINTS"
        or int(seal.get("n_jobs", -1)) != 16
        or seal.get("target_truth_opened") is not False
        or seal.get("txpert_commit") != TXPERT_COMMIT
    ):
        raise PredictionFailure("invalid E201 family seal")
    records = seal.get("records", [])
    expected_jobs = {(target, seed) for target in TARGETS for seed in SEEDS}
    actual_jobs = {
        (record.get("target"), int(record.get("seed", -1))) for record in records
    }
    if (
        len(records) != 16
        or actual_jobs != expected_jobs
        or canonical_hash(records) != seal.get("records_sha256")
    ):
        raise PredictionFailure("E201 family seal records are incomplete or changed")
    matches = [
        record
        for record in records
        if record.get("target") == args.target
        and int(record.get("seed", -1)) == args.seed
    ]
    if len(matches) != 1:
        raise PredictionFailure("target/seed is not unique in family seal")
    checkpoint_record = matches[0][args.checkpoint_role]
    checkpoint_path = resolve_data_path(checkpoint_record["path"], data_root)
    if not checkpoint_path.is_file():
        raise PredictionFailure(f"missing sealed checkpoint: {checkpoint_path}")
    if (
        checkpoint_path.stat().st_size != checkpoint_record["bytes"]
        or sha256_file(checkpoint_path) != checkpoint_record["sha256"]
    ):
        raise PredictionFailure("sealed checkpoint changed")

    blind_cache = txpert_repo / "cache/E201_prediction_blind"
    blind_h5ad = blind_cache / "de_adata_test.h5ad"
    blind_manifest_path = blind_cache / "E201_BLIND_PREDICTION_VIEW_MANIFEST.json"
    if (
        not blind_h5ad.is_file()
        or blind_h5ad.stat().st_size != 140_792_831
        or sha256_file(blind_h5ad) != BLIND_H5AD_SHA256
        or not blind_manifest_path.is_file()
        or sha256_file(blind_manifest_path) != BLIND_MANIFEST_SHA256
    ):
        raise PredictionFailure("blind target-prediction cache changed")
    blind_manifest = json.loads(blind_manifest_path.read_text(encoding="utf-8"))
    if (
        blind_manifest.get("status") != "BLIND_PREDICTION_VIEW_READY"
        or blind_manifest.get("source_h5ad_sha256") != OFFICIAL_SOURCE_H5AD_SHA256
        or int(blind_manifest.get("n_rows", -1)) != 581_172
        or int(blind_manifest.get("n_genes", -1)) != 3_352
        or int(blind_manifest.get("n_controls", -1)) != 39_165
        or int(blind_manifest.get("n_perturbed_rows", -1)) != 542_007
        or int(blind_manifest.get("perturbed_matrix_nonzero_values", -1)) != 0
        or blind_manifest.get("uns_keys") != []
    ):
        raise PredictionFailure("blind target-prediction manifest contract failed")
    if args.seed == 1:
        if shared_dir.exists():
            raise PredictionFailure(f"seed 1 refuses existing shared dir: {shared_dir}")
    elif not (shared_dir / "E201_SHARED_TARGET_MANIFEST.json").is_file():
        raise PredictionFailure(
            "seed 2-4 require the sealed seed-1 shared target files"
        )

    output_dir.mkdir(parents=True)
    status_path = output_dir / "E201_PREDICTION_RUN.json"
    metadata = {
        "experiment": "E201_txpert_multitarget_retraining",
        "status": "RUNNING",
        "started_at": now(),
        "target": args.target,
        "seed": args.seed,
        "checkpoint_role": args.checkpoint_role,
        "checkpoint_path": checkpoint_record["path"],
        "checkpoint_sha256": checkpoint_record["sha256"],
        "family_seal_path": seal_relpath,
        "family_seal_sha256": sha256_file(family_seal_path),
        "family_records_sha256": seal["records_sha256"],
        "safeconf_commit": safeconf_head,
        "safeconf_branch": safeconf_branch,
        "safeconf_tracking_commits": tracking,
        "prediction_adapter_path": script_relpath,
        "prediction_adapter_sha256": sha256_file(script_path),
        "txpert_commit": TXPERT_COMMIT,
        "official_source_h5ad_sha256": OFFICIAL_SOURCE_H5AD_SHA256,
        "blind_prediction_h5ad_sha256": BLIND_H5AD_SHA256,
        "blind_prediction_manifest_sha256": BLIND_MANIFEST_SHA256,
        "target_truth_materialized": False,
        "batch_size": args.batch_size,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "command": sys.argv,
    }
    write_json(status_path, metadata)

    prediction_partial = output_dir / "predictions.partial.npy"
    prediction_final = output_dir / "predictions.npy"
    try:
        os.chdir(txpert_repo)
        sys.path.insert(0, str(txpert_repo))

        import numpy as np
        import torch
        from omegaconf import OmegaConf

        from gspp.data.datamodule import PertDataModule
        from gspp.data.graphmodule import GSPGraph
        from gspp.predictor import PertPredictor
        from gspp.utils import set_seed

        source_contexts = sorted(set(TARGETS) - {args.target})
        cfg = compose(
            txpert_repo,
            [
                "mode=predict",
                "datamodule.task_type=E201_prediction_blind",
                "datamodule.train_cell_types=[" + ",".join(source_contexts) + "]",
                f"datamodule.test_cell_type={args.target}",
                "datamodule.val_cell_type=null",
                f"datamodule.batch_size={args.batch_size}",
                f"seed={args.seed}",
            ],
        )
        if (
            cfg.datamodule.match_cntr is not True
            or cfg.datamodule.avg_cntr is not True
            or cfg.datamodule.obsm_key != "raw"
        ):
            raise PredictionFailure("unexpected target control construction")
        set_seed(args.seed)
        if not torch.cuda.is_available():
            raise PredictionFailure("CUDA is required for E201 prediction")
        datamodule = PertDataModule(
            **OmegaConf.to_container(cfg.datamodule, resolve=True)
        )
        datamodule.prepare_data()
        datamodule.setup("test")
        if set(datamodule.test_adata.obs.cell_line.astype(str)) != {args.target}:
            raise PredictionFailure("test dataloader contains another cell line")
        graph = GSPGraph(
            pert2id=datamodule.pert2id,
            gene2id=datamodule.gene2id,
            **cfg.graph,
        )
        model = PertPredictor.load_from_checkpoint(
            checkpoint_path,
            input_dim=datamodule.input_dim,
            output_dim=datamodule.output_dim,
            adata_output_dim=datamodule.adata_output_dim,
            num_perts=len(datamodule.pert2id),
            graph=graph,
            pert_names=list(datamodule.pert2id.keys()),
            model_args=OmegaConf.to_container(cfg.model, resolve=True),
            device="cuda",
        ).to("cuda")
        model.eval()
        n_samples = len(datamodule.test_data)
        n_genes = datamodule.adata_output_dim
        predictions = np.lib.format.open_memmap(
            prediction_partial,
            mode="w+",
            dtype=np.float32,
            shape=(n_samples, n_genes),
        )

        shared_manifest_path = shared_dir / "E201_SHARED_TARGET_MANIFEST.json"
        if args.seed == 1:
            shared_dir.mkdir(parents=True)
            control_partial = shared_dir / "controls.partial.npy"
            obs_partial = shared_dir / "observations.partial.csv"
            controls = np.lib.format.open_memmap(
                control_partial,
                mode="w+",
                dtype=np.float32,
                shape=(n_samples, n_genes),
            )
            obs_handle = obs_partial.open("w", encoding="utf-8", newline="")
            obs_writer = csv.writer(obs_handle)
            obs_writer.writerow(
                ("row_index", "pert_cond_name", "cell_type", "experimental_batch")
            )
            shared_manifest = None
        else:
            shared_manifest = json.loads(
                shared_manifest_path.read_text(encoding="utf-8")
            )
            if (
                shared_manifest.get("status") != "SEALED_PRETRUTH_TARGET_INPUTS"
                or shared_manifest.get("target") != args.target
                or int(shared_manifest.get("seed_used_to_materialize", -1)) != 1
                or int(shared_manifest.get("n_samples", -1)) != n_samples
                or int(shared_manifest.get("n_genes", -1)) != n_genes
                or shared_manifest.get("blind_prediction_h5ad_sha256")
                != BLIND_H5AD_SHA256
                or shared_manifest.get("target_truth_materialized") is not False
            ):
                raise PredictionFailure("shared target manifest mismatch")
            shared_files = verify_shared_files(shared_dir, shared_manifest)
            controls = np.load(shared_files["controls"], mmap_mode="r")
            obs_handle = None
            obs_writer = None

        metadata_digest = hashlib.sha256()
        control_digest = hashlib.sha256()
        leakage = None
        target_expression_nonzero_values_seen = 0
        offset = 0
        started_predict = time.perf_counter()
        with torch.no_grad():
            for batch_index, batch in enumerate(datamodule.predict_dataloader()):
                batch_nonzero = int(torch.count_nonzero(batch.x).cpu())
                target_expression_nonzero_values_seen += batch_nonzero
                if batch_nonzero:
                    raise PredictionFailure(
                        "nonzero target expression entered pretruth prediction"
                    )
                prediction = model.sample_inference(
                    batch.control, batch.pert_idxs, batch.p, batch.cell_types
                )[:, :n_genes]
                if batch_index == 0:
                    # Hold control / perturbation tensors fixed. batch.control can
                    # share storage with batch.x; filling x in place would then
                    # corrupt the legitimate matched-control input and fail this
                    # check even when inference never reads target expression.
                    control_for_audit = batch.control.detach().clone()
                    pert_idxs_for_audit = batch.pert_idxs
                    p_for_audit = batch.p
                    cell_types_for_audit = batch.cell_types
                    aliased = bool(
                        batch.x.data_ptr() == batch.control.data_ptr()
                    )
                    batch.x.fill_(1.0)
                    prediction_after_dummy_change = model.sample_inference(
                        control_for_audit,
                        pert_idxs_for_audit,
                        p_for_audit,
                        cell_types_for_audit,
                    )[:, :n_genes]
                    batch.x.zero_()
                    leakage = {
                        "batch_x_aliased_with_control": aliased,
                        "prediction_exact_equal_after_dummy_x_change": bool(
                            torch.equal(prediction, prediction_after_dummy_change)
                        ),
                        "prediction_max_abs_delta_after_dummy_x_change": float(
                            torch.max(
                                torch.abs(prediction - prediction_after_dummy_change)
                            ).cpu()
                        ),
                    }
                    leakage["dummy_x_abs_tol"] = 1e-5
                    if (
                        leakage["prediction_max_abs_delta_after_dummy_x_change"]
                        > leakage["dummy_x_abs_tol"]
                    ):
                        raise PredictionFailure(
                            "prediction changed after dummy target-X modification "
                            f"(aliased={aliased}, "
                            f"max_abs_delta={leakage['prediction_max_abs_delta_after_dummy_x_change']})"
                        )
                pred_np = (
                    prediction.detach().cpu().numpy().astype(np.float32, copy=False)
                )
                control_np = (
                    batch.control[:, :n_genes]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
                end = offset + len(pred_np)
                predictions[offset:end] = pred_np
                if args.seed == 1:
                    controls[offset:end] = control_np
                else:
                    if not np.array_equal(controls[offset:end], control_np):
                        raise PredictionFailure(
                            "matched controls changed across seed runs"
                        )
                control_digest.update(array_bytes(control_np))
                for local_index, (condition, cell_type, experiment_batch) in enumerate(
                    zip(
                        batch.pert_cond_names,
                        batch.cell_types,
                        batch.experimental_batches,
                    )
                ):
                    row = (
                        offset + local_index,
                        str(condition),
                        str(cell_type),
                        str(experiment_batch),
                    )
                    metadata_digest.update(
                        (
                            json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                            + "\n"
                        ).encode("utf-8")
                    )
                    if obs_writer is not None:
                        obs_writer.writerow(row)
                offset = end
                if (batch_index + 1) % 500 == 0:
                    print(f"prediction batches complete: {batch_index + 1}", flush=True)
        predict_wall_seconds = time.perf_counter() - started_predict
        if offset != n_samples:
            raise PredictionFailure(f"prediction row count {offset} != {n_samples}")
        predictions.flush()
        del predictions
        os.replace(prediction_partial, prediction_final)

        if args.seed == 1:
            controls.flush()
            del controls
            obs_handle.close()
            control_final = shared_dir / "controls.npy"
            obs_final = shared_dir / "observations.csv"
            os.replace(control_partial, control_final)
            os.replace(obs_partial, obs_final)
            shared_manifest = {
                "experiment": "E201_txpert_multitarget_retraining",
                "status": "SEALED_PRETRUTH_TARGET_INPUTS",
                "created_at": now(),
                "target": args.target,
                "seed_used_to_materialize": 1,
                "n_samples": n_samples,
                "n_genes": n_genes,
                "blind_prediction_h5ad_sha256": BLIND_H5AD_SHA256,
                "target_truth_materialized": False,
                "target_expression_nonzero_values_seen": 0,
                "metadata_stream_sha256": metadata_digest.hexdigest(),
                "control_stream_sha256": control_digest.hexdigest(),
                "files": [
                    {
                        "role": "controls",
                        "path": "controls.npy",
                        "bytes": control_final.stat().st_size,
                        "sha256": sha256_file(control_final),
                    },
                    {
                        "role": "observations",
                        "path": "observations.csv",
                        "bytes": obs_final.stat().st_size,
                        "sha256": sha256_file(obs_final),
                    },
                ],
            }
            write_json(shared_manifest_path, shared_manifest)
        else:
            if metadata_digest.hexdigest() != shared_manifest["metadata_stream_sha256"]:
                raise PredictionFailure("target observation order changed across seeds")
            if control_digest.hexdigest() != shared_manifest["control_stream_sha256"]:
                raise PredictionFailure("control stream hash changed across seeds")

        metadata.update(
            {
                "status": "COMPLETE",
                "finished_at": now(),
                "n_samples": n_samples,
                "n_genes": n_genes,
                "n_prediction_batches": len(datamodule.predict_dataloader()),
                "predict_wall_seconds": predict_wall_seconds,
                "metadata_stream_sha256": metadata_digest.hexdigest(),
                "control_stream_sha256": control_digest.hexdigest(),
                "target_expression_nonzero_values_seen": (
                    target_expression_nonzero_values_seen
                ),
                "target_truth_materialized": False,
                "leakage_smoke": leakage,
                "prediction_file": {
                    "path": "predictions.npy",
                    "bytes": prediction_final.stat().st_size,
                    "sha256": sha256_file(prediction_final),
                },
                "shared_target_manifest_sha256": sha256_file(shared_manifest_path),
            }
        )
        write_json(status_path, metadata)
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
        write_json(status_path, metadata)
        raise


if __name__ == "__main__":
    main()
