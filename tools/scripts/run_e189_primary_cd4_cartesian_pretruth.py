#!/usr/bin/env python3
"""Train the frozen E189 scGPT–GEARS family without opening evaluation truth."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any
import uuid

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
BUILDER = ROOT / "tools/scripts/build_e189_primary_cd4_cartesian_assets.py"
HELPER_PATH = ROOT / "tools/scripts/run_e168_primary_cd4_pretruth.py"
E65_PATH = ROOT / "tools/scripts/run_e65_scgpt_formal_fixed_panel.py"
EVALUATOR = ROOT / "tools/scripts/run_e189_primary_cd4_cartesian_evaluation.py"
OUT = ROOT / "docs/实验结果/E189_primary_cd4_formal_cartesian_20260729"
DATA = Path("/home/yyf/data/safeconf_e189_primary_cd4_cartesian")
LOCKS = OUT / "MODEL_ASSET_LOCKS.csv"
CHECKPOINT_LOCKS = OUT / "SCGPT_CHECKPOINT_LOCKS.csv"
PREREG = (
    ROOT
    / "docs/实验结果/E188_advisor_aligned_experiment_program_20260729"
    / "PREREG_EXPERIMENT_PROGRAM.md"
)
PANELS = ("H01", "H02", "H03", "H04")
SUPPORT_LEVELS = (1, 2, 3, 5)
SEEDS = (3407, 3408, 3409)
N_GENES = 512
N_QUERIES = 840
EXPECTED_FILES = {
    "TRAIN_TASKS.csv",
    "VALIDATION_TASKS.csv",
    "QUERY_TASKS.csv",
    "TRAIN_EFFECTS.npz",
    "VALIDATION_EFFECTS.npz",
    "CONTROL_PROFILES.npz",
    "GENE_PANEL.csv",
    "TRAIN_NTC_COEXPRESSION_EDGES.csv",
    "GO_EDGES_PANEL.csv",
    "ASSET_MANIFEST.json",
}
FORBIDDEN_SUFFIXES = {".h5", ".h5ad", ".h5mu", ".loom", ".zarr"}


class ContractFailure(RuntimeError):
    """Fail-closed E189 data or code contract violation."""


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
    payload = path.read_bytes()
    try:
        committed = subprocess.check_output(
            ["git", "show", f"{head}:{relative}"], cwd=ROOT
        )
    except subprocess.CalledProcessError as exc:
        raise ContractFailure(f"formal input is not committed: {relative}") from exc
    if hashlib.sha256(payload).digest() != hashlib.sha256(committed).digest():
        raise ContractFailure(f"working file differs from HEAD: {relative}")
    return {"path": relative, "bytes": len(payload), "sha256": sha256_file(path)}


def verify_code_freeze(panel: str) -> tuple[str, str, dict[str, str], list[dict[str, Any]]]:
    head = git_text("rev-parse", "HEAD")
    branch = git_text("branch", "--show-current")
    if not branch:
        raise ContractFailure("formal run requires a named Git branch")
    remote_heads: dict[str, str] = {}
    for remote in ("origin", "github"):
        result = subprocess.run(
            [
                "git",
                "fetch",
                "--quiet",
                remote,
                f"refs/heads/{branch}:refs/remotes/{remote}/{branch}",
            ],
            cwd=ROOT,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            raise ContractFailure(
                f"cannot verify {remote}/{branch}: "
                f"{result.stderr.decode(errors='replace').strip()}"
            )
        remote_head = git_text("rev-parse", f"refs/remotes/{remote}/{branch}")
        if subprocess.run(
            ["git", "merge-base", "--is-ancestor", head, remote_head],
            cwd=ROOT,
            check=False,
        ).returncode:
            raise ContractFailure(f"HEAD {head} is absent from {remote}/{branch}")
        remote_heads[remote] = remote_head

    frozen = [
        RUNNER,
        BUILDER,
        HELPER_PATH,
        E65_PATH,
        EVALUATOR,
        PREREG,
        LOCKS,
        CHECKPOINT_LOCKS,
        OUT / "ASSET_BUILD_STATUS.json",
        OUT / "manifests/E189_ALL_TARGET_SPLITS.csv",
        OUT / f"manifests/E189_{panel}_TARGET_SPLIT.csv",
        OUT / f"manifests/E189_{panel}_QUERY_TASKS.csv",
    ]
    return head, branch, remote_heads, [require_committed(path, head) for path in frozen]


def import_helper() -> Any:
    spec = importlib.util.spec_from_file_location("e189_e168_helper", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise ContractFailure("cannot import frozen E168 model helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_npz(path: Path) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as archive:
        for key in archive.files:
            value = np.asarray(archive[key], dtype=np.float32)
            if value.shape != (N_GENES,) or not np.isfinite(value).all():
                raise ContractFailure(f"invalid vector {key} in {path.name}")
            result[str(key)] = value
    return result


def validate_assets(panel: str, support: int) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    package = (DATA / "model_assets" / panel / f"support_{support}").resolve(strict=True)
    expected_root = (DATA / "model_assets").resolve(strict=True)
    if package.parent.parent != expected_root or package.is_symlink():
        raise ContractFailure("asset package path is outside the frozen model-assets root")
    observed = {path.name for path in package.iterdir() if path.is_file()}
    directories = [path.name for path in package.iterdir() if path.is_dir()]
    if observed != EXPECTED_FILES or directories:
        raise ContractFailure(
            f"exact asset allowlist failed: files={sorted(observed)}, dirs={directories}"
        )
    for path in package.rglob("*"):
        lowered = path.as_posix().lower()
        if (
            path.suffix.lower() in FORBIDDEN_SUFFIXES
            or "evaluation_truth" in lowered
            or "f3_calibration" in lowered
            or "f4_evaluation" in lowered
        ):
            raise ContractFailure(f"truth/raw path entered model package: {path}")

    locks = pd.read_csv(LOCKS, keep_default_na=False)
    selected = locks.loc[
        locks.panel_id.astype(str).eq(panel)
        & locks.support.astype(int).eq(support)
    ].copy()
    if len(selected) != len(EXPECTED_FILES):
        raise ContractFailure("asset lock row count failed")
    input_hashes: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        path = DATA / str(row.path)
        if path.parent != package or path.name not in EXPECTED_FILES:
            raise ContractFailure(f"unsafe lock path: {row.path}")
        observed_sha = sha256_file(path)
        if observed_sha != str(row.sha256) or path.stat().st_size != int(row.bytes):
            raise ContractFailure(f"asset hash/size mismatch: {row.path}")
        input_hashes.append(
            {"path": str(row.path), "bytes": int(row.bytes), "sha256": observed_sha}
        )

    manifest = json.loads((package / "ASSET_MANIFEST.json").read_text())
    expected_manifest = {
        "experiment": "E189",
        "panel_id": panel,
        "support_contexts_per_seen_perturbation": support,
        "n_train_tasks": 120 * support,
        "n_validation_tasks": 360,
        "n_query_tasks": N_QUERIES,
        "n_seen_perturbations": 120,
        "n_unseen_perturbations": 40,
        "contains_evaluation_truth": False,
        "split_selection_uses_effect_values": False,
    }
    for key, value in expected_manifest.items():
        if manifest.get(key) != value:
            raise ContractFailure(f"asset manifest mismatch: {key}")
    return package, manifest, input_hashes


def validate_scgpt_checkpoint() -> list[dict[str, Any]]:
    locks = pd.read_csv(CHECKPOINT_LOCKS, keep_default_na=False)
    if set(Path(value).name for value in locks.path.astype(str)) != {
        "args.json",
        "vocab.json",
        "best_model.pt",
    }:
        raise ContractFailure("scGPT checkpoint lock set changed")
    hashes: list[dict[str, Any]] = []
    for row in locks.itertuples(index=False):
        path = Path(str(row.path))
        if path.is_symlink() or not path.is_file():
            raise ContractFailure(f"missing or symlinked scGPT checkpoint input: {path}")
        observed = sha256_file(path)
        if observed != str(row.sha256) or path.stat().st_size != int(row.bytes):
            raise ContractFailure(f"scGPT checkpoint hash/size mismatch: {path.name}")
        hashes.append(
            {"path": str(path), "bytes": int(row.bytes), "sha256": observed}
        )
    return hashes


def build_graphs(package: Path, support: int) -> tuple[
    dict[str, list[Any]], list[Any], pd.DataFrame, list[str], pd.DataFrame
]:
    import torch
    from torch_geometric.data import Data

    panel = pd.read_csv(package / "GENE_PANEL.csv", keep_default_na=False)
    panel = panel.sort_values("panel_index").reset_index(drop=True)
    if (
        len(panel) != N_GENES
        or not np.array_equal(panel.panel_index.to_numpy(int), np.arange(N_GENES))
        or panel.ensembl_id.astype(str).duplicated().any()
    ):
        raise ContractFailure("gene panel order/schema failed")

    train = pd.read_csv(package / "TRAIN_TASKS.csv", keep_default_na=False)
    validation = pd.read_csv(package / "VALIDATION_TASKS.csv", keep_default_na=False)
    query_frame = pd.read_csv(package / "QUERY_TASKS.csv", keep_default_na=False)
    train_effects = load_npz(package / "TRAIN_EFFECTS.npz")
    validation_effects = load_npz(package / "VALIDATION_EFFECTS.npz")
    controls = load_npz(package / "CONTROL_PROFILES.npz")
    coexpression = pd.read_csv(package / "TRAIN_NTC_COEXPRESSION_EDGES.csv")

    train_ids = set(train.task_id.astype(str))
    validation_ids = set(validation.task_id.astype(str))
    query_ids = set(query_frame.task_id.astype(str))
    if (
        len(train) != 120 * support
        or len(validation) != 360
        or len(query_frame) != N_QUERIES
        or set(train_effects) != train_ids
        or set(validation_effects) != validation_ids
        or train_ids & validation_ids
        or train_ids & query_ids
        or validation_ids & query_ids
    ):
        raise ContractFailure("task counts, effect keys, or exact task isolation failed")
    expected_settings = {
        "random_missing_pair": 120,
        "unseen_context_row": 360,
        "unseen_perturbation_column": 240,
        "double_unseen": 120,
    }
    if query_frame.e189_setting.value_counts().to_dict() != expected_settings:
        raise ContractFailure("query setting counts changed")
    supervised_genes = set(train.perturbed_gene_id.astype(str)) | set(
        validation.perturbed_gene_id.astype(str)
    )
    unseen_genes = set(
        query_frame.loc[
            query_frame.e189_seen_perturbation.astype(str).str.lower().eq("false"),
            "perturbed_gene_id",
        ].astype(str)
    )
    if len(supervised_genes) != 120 or len(unseen_genes) != 40 or supervised_genes & unseen_genes:
        raise ContractFailure("unseen perturbation entered supervision")

    targets = set(train.perturbed_gene_id.astype(str)) | set(
        validation.perturbed_gene_id.astype(str)
    ) | set(query_frame.perturbed_gene_id.astype(str))
    target_position = {
        str(row.ensembl_id): int(row.panel_index)
        for row in panel.itertuples(index=False)
        if str(row.ensembl_id) in targets
    }
    if len(target_position) != 160:
        raise ContractFailure("not all 160 E189 targets occur in the gene panel")

    supervised: dict[str, list[Any]] = {"train": [], "validation": []}
    audit: list[dict[str, Any]] = []
    for role, frame, effects in (
        ("train", train, train_effects),
        ("validation", validation, validation_effects),
    ):
        for row in frame.itertuples(index=False):
            task_id = str(row.task_id)
            basal = controls[f"{row.donor_id}::{row.culture_condition}"]
            flag = np.zeros(N_GENES, np.float32)
            flag[target_position[str(row.perturbed_gene_id)]] = 1.0
            graph = Data(
                x=torch.from_numpy(np.stack([basal, flag], axis=1)),
                y=torch.from_numpy(basal + effects[task_id]).unsqueeze(0),
                pert=task_id,
                donor_id=str(row.donor_id),
                culture_condition=str(row.culture_condition),
                perturbed_gene_id=str(row.perturbed_gene_id),
                split=role,
            )
            supervised[role].append(graph)
            audit.append(
                {"task_id": task_id, "graph_role": f"supervised_{role}", "contains_y": True}
            )

    query: list[Any] = []
    for row in query_frame.itertuples(index=False):
        basal = controls[f"{row.donor_id}::{row.culture_condition}"]
        flag = np.zeros(N_GENES, np.float32)
        flag[target_position[str(row.perturbed_gene_id)]] = 1.0
        graph = Data(
            x=torch.from_numpy(np.stack([basal, flag], axis=1)),
            pert=str(row.task_id),
            donor_id=str(row.donor_id),
            culture_condition=str(row.culture_condition),
            perturbed_gene_id=str(row.perturbed_gene_id),
            split="query",
            e189_setting=str(row.e189_setting),
        )
        if getattr(graph, "y", None) is not None:
            raise ContractFailure("query graph contains y")
        query.append(graph)
        audit.append(
            {"task_id": str(row.task_id), "graph_role": "query", "contains_y": False}
        )
    if len(query) != N_QUERIES:
        raise ContractFailure("query graph count failed")
    genes = panel.scgpt_token.astype(str).tolist()
    return supervised, query, pd.DataFrame(audit), genes, coexpression


def prediction_matrix(
    predictions: dict[str, np.ndarray], query_order: list[str], model_name: str
) -> tuple[np.ndarray, dict[str, Any]]:
    if set(predictions) != set(query_order) or len(predictions) != N_QUERIES:
        raise ContractFailure(f"{model_name}: prediction IDs do not match queries")
    matrix = np.stack([predictions[task_id] for task_id in query_order]).astype(np.float32)
    if matrix.shape != (N_QUERIES, N_GENES) or not np.isfinite(matrix).all():
        raise ContractFailure(f"{model_name}: prediction matrix invalid")
    quantized = np.rint(matrix / 1e-6).astype(np.int64)
    fingerprints = {hashlib.sha256(row.tobytes()).digest() for row in quantized}
    max_coordinate_std = float(np.max(np.std(matrix, axis=0)))
    if len(fingerprints) < 2 or max_coordinate_std <= 1e-6:
        raise ContractFailure(f"{model_name}: task-independent prediction collapse")
    return matrix, {
        "model_key": model_name,
        "n_query_predictions": len(matrix),
        "n_unique_vectors_at_1e-6": len(fingerprints),
        "max_coordinate_std": max_coordinate_std,
        "all_finite": True,
        "query_graphs_containing_y": 0,
    }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run(panel: str, support: int, device_name: str) -> dict[str, Any]:
    started = time.time()
    release = OUT / "pretruth_release" / panel / f"support_{support}"
    staging = OUT / ".pretruth_staging" / panel / f"support_{support}"
    if release.exists() or staging.exists():
        raise ContractFailure(f"append-only pretruth release already exists: {panel}/support_{support}")
    head, branch, remote_heads, code_hashes = verify_code_freeze(panel)
    package, manifest, asset_hashes = validate_assets(panel, support)
    checkpoint_hashes = validate_scgpt_checkpoint()
    supervised, query, graph_audit, genes, coexpression = build_graphs(package, support)

    import torch

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ContractFailure(f"requested CUDA device is unavailable: {device_name}")
    helper = import_helper()
    helper.GO_FILE = package / "GO_EDGES_PANEL.csv"
    query_order = [str(graph.pert) for graph in query]
    arrays: dict[str, np.ndarray] = {}
    histories: list[pd.DataFrame] = []
    model_audits: list[dict[str, Any]] = []
    for seed in SEEDS:
        for architecture in ("scGPT", "GEARS"):
            key = f"{architecture}_seed{seed}"
            if architecture == "scGPT":
                predictions, history, fit_audit = helper.train_scgpt(
                    seed, supervised, query, genes, device
                )
            else:
                predictions, history, fit_audit = helper.train_gears(
                    seed, supervised, query, genes, coexpression, device
                )
            arrays[key], prediction_audit = prediction_matrix(
                predictions, query_order, key
            )
            histories.append(history)
            model_audits.append(
                {
                    "panel_id": panel,
                    "support": support,
                    "architecture": architecture,
                    "seed": seed,
                    **fit_audit,
                    **prediction_audit,
                }
            )
            del predictions
            if device.type == "cuda":
                torch.cuda.empty_cache()

    staging.mkdir(parents=True, exist_ok=False)
    (staging / "arrays").mkdir()
    (staging / "tables").mkdir()
    with (staging / "arrays/PRETRUTH_PREDICTIONS.npz").open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    pd.DataFrame({"query_index": range(N_QUERIES), "task_id": query_order}).to_csv(
        staging / "tables/QUERY_ORDER.csv", index=False
    )
    graph_audit.to_csv(staging / "tables/GRAPH_AUDIT.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(
        staging / "tables/TRAINING_HISTORY.csv", index=False
    )
    pd.DataFrame(model_audits).to_csv(
        staging / "tables/MODEL_AUDIT.csv", index=False
    )
    pd.DataFrame(code_hashes + asset_hashes + checkpoint_hashes).to_csv(
        staging / "tables/INPUT_HASHES.csv", index=False
    )
    runtime = {
        "experiment": "E189",
        "stage": "PRETRUTH_PREDICTION",
        "status": "PASS",
        "panel_id": panel,
        "support_contexts_per_seen_perturbation": support,
        "n_train_tasks": manifest["n_train_tasks"],
        "n_validation_tasks": manifest["n_validation_tasks"],
        "n_query_tasks": N_QUERIES,
        "model_family_members": sorted(arrays),
        "evaluation_truth_files_read": 0,
        "query_graphs_containing_y": 0,
        "git_head": head,
        "git_branch": branch,
        "code_freeze_remote_heads": remote_heads,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "wall_seconds": time.time() - started,
    }
    atomic_json(staging / "PRETRUTH_STATUS.json", runtime)
    files = sorted(path for path in staging.rglob("*") if path.is_file())
    pd.DataFrame(
        [
            {
                "path": path.relative_to(staging).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ]
    ).to_csv(staging / "RELEASE_LOCKS.csv", index=False)
    release.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, release)
    return {
        "status": "PASS",
        "release": release.relative_to(ROOT).as_posix(),
        "panel": panel,
        "support": support,
        "wall_seconds": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", required=True, choices=PANELS)
    parser.add_argument("--support", required=True, type=int, choices=SUPPORT_LEVELS)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = run(args.panel, args.support, args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
