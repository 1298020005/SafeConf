#!/usr/bin/env python3
"""Train E190 in Adamson and predict Replogle queries without target truth."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import uuid

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
BUILDER = ROOT / "tools/scripts/build_e190_adamson_replogle_pretruth_assets.py"
HELPER_PATH = ROOT / "tools/scripts/run_e168_primary_cd4_pretruth.py"
E65_PATH = ROOT / "tools/scripts/run_e65_scgpt_formal_fixed_panel.py"
OUT = ROOT / "docs/实验结果/E190_adamson_to_replogle_direct_transfer_20260729"
DATA = Path("/home/yyf/data/safeconf_e190_adamson_replogle")
ASSETS = DATA / "model_assets"
RELEASE = OUT / "pretruth_release"
STAGING = OUT / ".pretruth_staging"
LOCKS = OUT / "MODEL_ASSET_LOCKS.csv"
CHECKPOINT_LOCKS = OUT / "SCGPT_CHECKPOINT_LOCKS.csv"
N_GENES = 512
N_QUERIES = 692
SEEDS = (3407, 3408, 3409)
EXPECTED_FILES = {
    "ASSET_MANIFEST.json",
    "GENE_PANEL.csv",
    "GO_EDGES_PANEL.csv",
    "QUERY_TASKS.csv",
    "SOURCE_CONTROL_COEXPRESSION_EDGES.csv",
    "SOURCE_CONTROL_PROFILE.npz",
    "SOURCE_GENE_EFFECTS.npz",
    "TARGET_CONTROL_PROFILES.npz",
    "TRAIN_EFFECTS.npz",
    "TRAIN_TASKS.csv",
    "VALIDATION_EFFECTS.npz",
    "VALIDATION_TASKS.csv",
}


class ContractFailure(RuntimeError):
    """Fail-closed E190 pretruth contract error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_text(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def require_committed(path: Path, head: str) -> dict[str, Any]:
    relative = path.relative_to(ROOT).as_posix()
    committed = subprocess.check_output(["git", "show", f"{head}:{relative}"], cwd=ROOT)
    if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(committed).digest():
        raise ContractFailure(f"working file differs from HEAD: {relative}")
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def verify_freeze() -> tuple[str, str, dict[str, str], list[dict[str, Any]]]:
    head = git_text("rev-parse", "HEAD")
    branch = git_text("branch", "--show-current")
    if not branch:
        raise ContractFailure("E190 formal run requires a named branch")
    remote_heads = {}
    for remote in ("origin", "github"):
        subprocess.run(
            [
                "git",
                "fetch",
                "--quiet",
                remote,
                f"refs/heads/{branch}:refs/remotes/{remote}/{branch}",
            ],
            cwd=ROOT,
            check=True,
        )
        remote_head = git_text("rev-parse", f"refs/remotes/{remote}/{branch}")
        if subprocess.run(
            ["git", "merge-base", "--is-ancestor", head, remote_head],
            cwd=ROOT,
            check=False,
        ).returncode:
            raise ContractFailure(f"HEAD absent from {remote}/{branch}")
        remote_heads[remote] = remote_head
    frozen = [
        RUNNER,
        BUILDER,
        HELPER_PATH,
        E65_PATH,
        OUT / "PREREG_ANALYSIS_PLAN.md",
        OUT / "PRETRUTH_SCALE_AMENDMENT.md",
        OUT / "METADATA_FREEZE_STATUS.json",
        OUT / "E190_QUERY_MANIFEST.csv",
        OUT / "E190_SOURCE_CELL_FOLD_ASSIGNMENTS.csv",
        OUT / "E190_TARGET_CONTROL_ASSIGNMENTS.csv",
        OUT / "ASSET_BUILD_STATUS.json",
        OUT / "MODEL_ASSET_LOCKS.csv",
        OUT / "SCGPT_CHECKPOINT_LOCKS.csv",
        OUT / "RAW_SOURCE_LOCKS.json",
        OUT / "PRETRUTH_INPUT_SCALE_AUDIT.csv",
    ]
    return head, branch, remote_heads, [require_committed(path, head) for path in frozen]


def load_npz(path: Path) -> dict[str, np.ndarray]:
    result = {}
    with np.load(path, allow_pickle=False) as archive:
        for key in archive.files:
            value = np.asarray(archive[key], np.float32)
            if value.shape != (N_GENES,) or not np.isfinite(value).all():
                raise ContractFailure(f"invalid vector {key} in {path.name}")
            result[str(key)] = value
    return result


def validate_assets() -> list[dict[str, Any]]:
    root = ASSETS.resolve(strict=True)
    if root.is_symlink() or root != (DATA / "model_assets").resolve():
        raise ContractFailure("unsafe E190 model asset root")
    observed = {path.name for path in root.iterdir() if path.is_file()}
    if observed != EXPECTED_FILES or any(path.is_dir() for path in root.iterdir()):
        raise ContractFailure("E190 exact asset allowlist failed")
    if any(
        token in path.as_posix().lower()
        for path in root.rglob("*")
        for token in ("evaluation_truth", "target_true", "target_effect")
    ):
        raise ContractFailure("target evaluation truth entered E190 model assets")
    locks = pd.read_csv(LOCKS, keep_default_na=False)
    if len(locks) != len(EXPECTED_FILES):
        raise ContractFailure("E190 asset lock count failed")
    hashes = []
    for row in locks.itertuples(index=False):
        path = DATA / str(row.path)
        if path.parent.resolve() != root or path.name not in EXPECTED_FILES:
            raise ContractFailure(f"unsafe E190 asset lock path: {row.path}")
        observed_sha = sha256_file(path)
        if observed_sha != str(row.sha256) or path.stat().st_size != int(row.bytes):
            raise ContractFailure(f"E190 asset hash/size mismatch: {row.path}")
        hashes.append(
            {"path": str(row.path), "bytes": int(row.bytes), "sha256": observed_sha}
        )
    manifest = json.loads((root / "ASSET_MANIFEST.json").read_text())
    required = {
        "experiment": "E190",
        "stage": "PRETRUTH_MODEL_ASSET_BUILD",
        "status": "PASS",
        "n_panel_genes": 512,
        "n_transfer_genes": 47,
        "n_train_tasks": 216,
        "n_validation_tasks": 54,
        "n_target_queries": 692,
        "n_target_control_profiles": 48,
        "target_perturbation_x_rows_read": 0,
        "contains_target_evaluation_truth": False,
        "normalization": "per-cell full-library scale to 10000, then log1p",
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ContractFailure(f"E190 asset manifest mismatch: {key}")
    return hashes


def validate_checkpoint() -> list[dict[str, Any]]:
    locks = pd.read_csv(CHECKPOINT_LOCKS, keep_default_na=False)
    hashes = []
    for row in locks.itertuples(index=False):
        path = Path(str(row.path))
        observed = sha256_file(path)
        if observed != str(row.sha256) or path.stat().st_size != int(row.bytes):
            raise ContractFailure(f"scGPT checkpoint mismatch: {path.name}")
        hashes.append(
            {"path": str(path), "bytes": int(row.bytes), "sha256": observed}
        )
    return hashes


def import_helper() -> Any:
    spec = importlib.util.spec_from_file_location("e190_e168_helper", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise ContractFailure("cannot import frozen E168 model helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_graphs() -> tuple[
    dict[str, list[Any]], list[Any], pd.DataFrame, list[str], pd.DataFrame
]:
    import torch
    from torch_geometric.data import Data

    panel = pd.read_csv(ASSETS / "GENE_PANEL.csv", keep_default_na=False)
    panel = panel.sort_values("panel_index").reset_index(drop=True)
    if (
        len(panel) != N_GENES
        or not np.array_equal(panel.panel_index.to_numpy(int), np.arange(N_GENES))
        or panel.scgpt_token.astype(str).duplicated().any()
    ):
        raise ContractFailure("E190 gene panel order/token uniqueness failed")
    target_position = {
        str(row.gene_name): int(row.panel_index)
        for row in panel.itertuples(index=False)
        if str(row.panel_role) == "TRANSFER_TARGET"
    }
    if len(target_position) != 47:
        raise ContractFailure("E190 transfer target positions changed")

    train = pd.read_csv(ASSETS / "TRAIN_TASKS.csv", keep_default_na=False)
    validation = pd.read_csv(ASSETS / "VALIDATION_TASKS.csv", keep_default_na=False)
    query_frame = pd.read_csv(ASSETS / "QUERY_TASKS.csv", keep_default_na=False)
    train_effects = load_npz(ASSETS / "TRAIN_EFFECTS.npz")
    validation_effects = load_npz(ASSETS / "VALIDATION_EFFECTS.npz")
    source_control = load_npz(ASSETS / "SOURCE_CONTROL_PROFILE.npz")
    target_controls = load_npz(ASSETS / "TARGET_CONTROL_PROFILES.npz")
    coexpression = pd.read_csv(ASSETS / "SOURCE_CONTROL_COEXPRESSION_EDGES.csv")
    if (
        len(train) != 216
        or len(validation) != 54
        or len(query_frame) != N_QUERIES
        or set(train.task_id.astype(str)) != set(train_effects)
        or set(validation.task_id.astype(str)) != set(validation_effects)
        or query_frame.task_id.duplicated().any()
    ):
        raise ContractFailure("E190 task/effect counts changed")
    query_ids = set(query_frame.task_id.astype(str))
    if query_ids & (set(train_effects) | set(validation_effects)):
        raise ContractFailure("E190 target query ID entered source supervision")

    supervised: dict[str, list[Any]] = {"train": [], "validation": []}
    audit = []
    for role, frame, effects in (
        ("train", train, train_effects),
        ("validation", validation, validation_effects),
    ):
        for row in frame.itertuples(index=False):
            basal = source_control[str(row.context_key)]
            flag = np.zeros(N_GENES, np.float32)
            flag[target_position[str(row.gene)]] = 1.0
            graph = Data(
                x=torch.from_numpy(np.stack([basal, flag], axis=1)),
                y=torch.from_numpy(basal + effects[str(row.task_id)]).unsqueeze(0),
                pert=str(row.task_id),
                perturbed_gene_id=str(row.gene),
                split=role,
            )
            supervised[role].append(graph)
            audit.append(
                {"task_id": str(row.task_id), "graph_role": f"supervised_{role}", "contains_y": True}
            )

    query = []
    for row in query_frame.itertuples(index=False):
        basal = target_controls[str(row.context_key)]
        flag = np.zeros(N_GENES, np.float32)
        flag[target_position[str(row.gene)]] = 1.0
        graph = Data(
            x=torch.from_numpy(np.stack([basal, flag], axis=1)),
            pert=str(row.task_id),
            perturbed_gene_id=str(row.gene),
            target_batch=str(row.batch),
            split="target_query",
        )
        if getattr(graph, "y", None) is not None:
            raise ContractFailure("E190 target query graph contains y")
        query.append(graph)
        audit.append(
            {"task_id": str(row.task_id), "graph_role": "target_query", "contains_y": False}
        )
    genes = panel.scgpt_token.astype(str).tolist()
    return supervised, query, pd.DataFrame(audit), genes, coexpression


def prediction_matrix(
    predictions: dict[str, np.ndarray], order: list[str], key: str
) -> tuple[np.ndarray, dict[str, Any]]:
    if set(predictions) != set(order):
        raise ContractFailure(f"{key}: prediction IDs differ from target queries")
    matrix = np.stack([predictions[task_id] for task_id in order]).astype(np.float32)
    if matrix.shape != (N_QUERIES, N_GENES) or not np.isfinite(matrix).all():
        raise ContractFailure(f"{key}: invalid prediction matrix")
    encoded = np.rint(matrix / 1e-6).astype(np.int64)
    unique = len({hashlib.sha256(row.tobytes()).digest() for row in encoded})
    if unique < 2 or float(np.max(np.std(matrix, axis=0))) <= 1e-6:
        raise ContractFailure(f"{key}: target predictions collapsed")
    return matrix, {
        "model_key": key,
        "n_predictions": len(matrix),
        "n_unique_vectors_at_1e-6": unique,
        "all_finite": True,
        "target_query_graphs_containing_y": 0,
    }


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run(device_name: str) -> dict[str, Any]:
    if RELEASE.exists() or STAGING.exists():
        raise ContractFailure("E190 pretruth release is append-only and already exists")
    started = time.time()
    head, branch, remote_heads, code_hashes = verify_freeze()
    asset_hashes = validate_assets()
    checkpoint_hashes = validate_checkpoint()
    supervised, query, graph_audit, genes, coexpression = build_graphs()

    import torch

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ContractFailure(f"requested device unavailable: {device_name}")
    helper = import_helper()
    helper.GO_FILE = ASSETS / "GO_EDGES_PANEL.csv"
    order = [str(graph.pert) for graph in query]
    arrays = {}
    histories = []
    audits = []
    for seed in SEEDS:
        for architecture in ("scGPT", "GEARS"):
            key = f"{architecture}_seed{seed}"
            if architecture == "scGPT":
                predictions, history, fit = helper.train_scgpt(
                    seed, supervised, query, genes, device
                )
            else:
                predictions, history, fit = helper.train_gears(
                    seed, supervised, query, genes, coexpression, device
                )
            arrays[key], prediction_audit = prediction_matrix(predictions, order, key)
            histories.append(history)
            audits.append(
                {
                    "architecture": architecture,
                    "seed": seed,
                    **fit,
                    **prediction_audit,
                }
            )
            del predictions
            if device.type == "cuda":
                torch.cuda.empty_cache()

    (STAGING / "arrays").mkdir(parents=True)
    (STAGING / "tables").mkdir()
    with (STAGING / "arrays/PRETRUTH_PREDICTIONS.npz").open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    pd.DataFrame({"query_index": range(N_QUERIES), "task_id": order}).to_csv(
        STAGING / "tables/QUERY_ORDER.csv", index=False
    )
    graph_audit.to_csv(STAGING / "tables/GRAPH_AUDIT.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(
        STAGING / "tables/TRAINING_HISTORY.csv", index=False
    )
    pd.DataFrame(audits).to_csv(STAGING / "tables/MODEL_AUDIT.csv", index=False)
    pd.DataFrame(code_hashes + asset_hashes + checkpoint_hashes).to_csv(
        STAGING / "tables/INPUT_HASHES.csv", index=False
    )
    status = {
        "experiment": "E190",
        "stage": "PRETRUTH_PREDICTION",
        "status": "PASS",
        "n_source_train_tasks": 216,
        "n_source_validation_tasks": 54,
        "n_target_queries": N_QUERIES,
        "target_perturbation_x_rows_read": 0,
        "target_query_graphs_containing_y": 0,
        "model_family_members": sorted(arrays),
        "git_head": head,
        "git_branch": branch,
        "code_freeze_remote_heads": remote_heads,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "wall_seconds": time.time() - started,
    }
    atomic_json(STAGING / "PRETRUTH_STATUS.json", status)
    files = sorted(path for path in STAGING.rglob("*") if path.is_file())
    pd.DataFrame(
        [
            {
                "path": path.relative_to(STAGING).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ]
    ).to_csv(STAGING / "RELEASE_LOCKS.csv", index=False)
    os.replace(STAGING, RELEASE)
    return {
        "status": "PASS",
        "release": RELEASE.relative_to(ROOT).as_posix(),
        "wall_seconds": time.time() - started,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    print(json.dumps(run(args.device), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
