#!/usr/bin/env python3
"""E157: train native PRESCRIBE on E156 development assets and lock task scores.

Formal execution is deliberately split from evaluation.  E156 contains no test
expression asset, and this runner refuses to proceed if a legacy
``sealed_test_transform.h5ad`` is present.  After a checkpoint is locked, each
frozen test perturbation is queried once with train-control expression and its
perturbation string; no held-out expression or label is present in those graphs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PRESCRIBE = Path("/home/yyf/archive/external/PRESCRIBE")
E155 = ROOT / "docs/实验结果/E155_prescribe_norman_p3p4_contract_20260714"
E156 = ROOT / "docs/实验结果/E156_prescribe_norman_p3p4_preprocess_20260714"
OUT = ROOT / "docs/实验结果/E157_prescribe_norman_p3p4_native_20260714"
CONTRACT = OUT / "ANALYSIS_CONTRACT.md"
LOCKED_MODEL_ROOT = Path("/home/yyf/data/safeconf_e157_locked_models")
SEED = 3407
PANEL_CONFIG = {
    "p3": {"panel": "Norman_P3", "data_name": "norman_p3", "physical_gpu": 0},
    "p4": {"panel": "Norman_P4", "data_name": "norman_p4", "physical_gpu": 1},
}
N_EPOCHS = 50
N_WARMUP_EPOCHS = 5
BATCH_SIZE = 4096
N_PCA = 10
N_EQUIVALENCE_TASKS = 8
GRAPH_X_ATOL = 1e-6
FORWARD_EQUIVALENCE_ATOL = 1e-5
EXPECTED_E156_PHASE = "complete_preprocessing_only_no_training_no_evaluation"
EXPECTED_PRESCRIBE_COMMIT = "6f7264a205aaff654a9594863c5c10b656f88ebe"
FROZEN_E155_SOURCES = {
    "PRESCRIBE_Step2_train": PRESCRIBE / "Step2_train.py",
    "PRESCRIBE_data_loader_worktree": PRESCRIBE / "src/data/pertdata.py",
    "PRESCRIBE_GEARS_data_worktree": PRESCRIBE / "gears/pertdata.py",
    "scGPT_perturbation_embedding": PRESCRIBE / "scLLM_weights/scGPT/embedding.pkl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", required=True, choices=sorted(PANEL_CONFIG))
    parser.add_argument("--mode", required=True, choices=["dry-run", "formal"])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--gpu-index", type=int)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_frozen_csv(path: Path, frame: pd.DataFrame) -> str:
    """Write once, then require byte-identical content on every recovery run."""
    payload = frame.to_csv(index=False).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    if path.exists():
        if sha256_file(path) != digest:
            raise RuntimeError(f"Frozen run manifest changed during recovery: {path}")
        return digest
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return digest


def formal_git_provenance() -> dict[str, object]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    records: dict[str, object] = {"runner_git_head": head}
    for role, path in {
        "runner": Path(__file__).resolve(),
        "contract": CONTRACT.resolve(),
    }.items():
        relative = path.relative_to(ROOT)
        try:
            committed = subprocess.check_output(
                ["git", "show", f"HEAD:{relative.as_posix()}"], cwd=ROOT
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Formal E157 {role} is not committed at HEAD") from exc
        committed_sha256 = hashlib.sha256(committed).hexdigest()
        working_sha256 = sha256_file(path)
        if committed_sha256 != working_sha256:
            raise RuntimeError(f"Formal E157 {role} differs from the committed HEAD blob")
        records[f"{role}_sha256"] = working_sha256
        records[f"{role}_matches_git_head_blob"] = True
    return records


def update_status(path: Path, **updates: object) -> dict[str, object]:
    current: dict[str, object] = {}
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
    current.update(updates)
    atomic_json(path, current)
    return current


def source_manifest() -> pd.DataFrame:
    files = {
        Path(__file__).resolve(),
        PRESCRIBE / "Step2_train.py",
        PRESCRIBE / "gears/pertdata.py",
        *PRESCRIBE.joinpath("src").rglob("*.py"),
    }
    return pd.DataFrame(
        [
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(files, key=str)
        ]
    )


def expected_e155_source_hash(role: str) -> str:
    table = pd.read_csv(E155 / "manifests/E155_SOURCE_HASHES.csv")
    rows = table.loc[table["source_role"].eq(role), "sha256"]
    if len(rows) != 1:
        raise RuntimeError(f"E155 source role is not unique: {role}")
    return str(rows.iloc[0])


def verify_e156_panel_assets(
    panel_dir: Path,
    known_hashes: dict[Path, str],
) -> dict[str, str]:
    manifest = pd.read_csv(E156 / "tables/E156_ARTIFACT_HASHES.csv")
    manifest["path"] = manifest["path"].astype(str)
    prefix = str(panel_dir.resolve()) + os.sep
    rows = manifest.loc[manifest["path"].str.startswith(prefix)]
    if rows.empty:
        raise RuntimeError(f"E156 manifest has no assets for {panel_dir}")
    verified: dict[str, str] = {}
    for record in rows.to_dict("records"):
        path = Path(str(record["path"]))
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = known_hashes.get(path.resolve())
        if actual is None:
            actual = sha256_file(path)
        expected = str(record["sha256"])
        if actual != expected:
            raise RuntimeError(f"E156 asset changed: {path}")
        verified[str(path)] = actual
    return verified


def input_manifest(inputs: dict[str, object]) -> pd.DataFrame:
    rows = [
        {"role": "E156_panel_asset", "path": path, "sha256": digest}
        for path, digest in sorted(inputs["e156_asset_hashes"].items())
    ]
    rows.extend(
        {"role": role, "path": str(FROZEN_E155_SOURCES[role]), "sha256": digest}
        for role, digest in sorted(inputs["fixed_source_hashes"].items())
    )
    for role, path in [
        ("E155_split_csv", Path(inputs["split_csv"])),
        ("E155_split_pickle", Path(inputs["split_pkl"])),
        ("E155_run_status", E155 / "RUN_STATUS.json"),
        ("E156_run_status", E156 / "RUN_STATUS.json"),
        ("E156_runner", Path(inputs["e156_status"]["runner_path"])),
        ("E156_artifact_manifest", E156 / "tables/E156_ARTIFACT_HASHES.csv"),
        ("E157_analysis_contract", CONTRACT),
    ]:
        rows.append({"role": role, "path": str(path), "sha256": sha256_file(path)})
    return pd.DataFrame(rows).sort_values(["role", "path"]).reset_index(drop=True)


def validate_inputs(config: dict[str, object]) -> dict[str, object]:
    e155_status = json.loads((E155 / "RUN_STATUS.json").read_text(encoding="utf-8"))
    e156_status = json.loads((E156 / "RUN_STATUS.json").read_text(encoding="utf-8"))
    if e156_status.get("phase") != EXPECTED_E156_PHASE:
        raise RuntimeError(f"E156 phase is not frozen preprocessing-complete: {e156_status.get('phase')}")
    if any(
        bool(e156_status.get(key))
        for key in ["model_training_started", "predictions_generated", "test_endpoint_computed"]
    ):
        raise RuntimeError("E156 already reports training/prediction/evaluation")
    for key in [
        "test_X_rows_indexed",
        "test_X_rows_materialized",
        "test_X_rows_transformed",
    ]:
        if e156_status.get(key) is not False:
            raise RuntimeError(f"E156 does not certify {key}=false")
    if e156_status.get("test_expression_transformed") is not False:
        raise RuntimeError("E156 reports a transformed test-expression asset")
    if e156_status.get("runner_matches_git_head_blob") is not True:
        raise RuntimeError("E156 runner was not frozen to its recorded Git HEAD")
    e156_runner = Path(str(e156_status["runner_path"]))
    if sha256_file(e156_runner) != str(e156_status["runner_sha256"]):
        raise RuntimeError("E156 runner changed after preprocessing")
    if e155_status.get("model_and_preprocess_seed") != SEED:
        raise RuntimeError("E155 seed mismatch")

    fixed_source_hashes = {}
    for role, path in FROZEN_E155_SOURCES.items():
        actual = sha256_file(path)
        if actual != expected_e155_source_hash(role):
            raise RuntimeError(f"Frozen E155 source changed: {role}")
        fixed_source_hashes[role] = actual
    if sha256_file(PRESCRIBE / "src/nn/loss.py") != e156_status["native_loss_sha256"]:
        raise RuntimeError("PRESCRIBE native loss changed after E156")
    if (
        sha256_file(PRESCRIBE / "src/model/lightening_module.py")
        != e156_status["native_lightning_module_sha256"]
    ):
        raise RuntimeError("PRESCRIBE native Lightning module changed after E156")

    manifest_path = E156 / "tables/E156_ARTIFACT_HASHES.csv"
    if sha256_file(manifest_path) != e156_status["artifact_manifest_sha256"]:
        raise RuntimeError("E156 artifact-manifest hash changed")
    panel = str(config["panel"])
    data_name = str(config["data_name"])
    panel_status = next(item for item in e156_status["panels"] if item["panel"] == panel)
    dev_h5ad = Path(panel_status["h5ad"])
    graph_path = Path(e156_status["graph_summaries"][[
        item["panel"] for item in e156_status["graph_summaries"]
    ].index(panel)]["cell_graphs_path"])
    dev_hash = sha256_file(dev_h5ad)
    graph_hash = sha256_file(graph_path)
    if dev_hash != e156_status["perturb_processed_sha256"][panel]:
        raise RuntimeError(f"{panel}: E156 development H5AD changed")
    if graph_hash != e156_status["cell_graphs_sha256"][panel]:
        raise RuntimeError(f"{panel}: E156 cell graphs changed")
    e156_asset_hashes = verify_e156_panel_assets(
        dev_h5ad.parent,
        {dev_h5ad.resolve(): dev_hash, graph_path.resolve(): graph_hash},
    )

    legacy_test_assets = sorted(
        str(path) for path in Path(e156_status["data_root"]).rglob("sealed_test_transform.h5ad")
    )
    if legacy_test_assets:
        raise RuntimeError(f"Legacy E156 test-expression assets are present: {legacy_test_assets}")

    split_csv = E155 / "manifests" / f"{panel}_SPLIT.csv"
    split_pkl = E155 / "manifests" / f"{panel}_set2conditions.pkl"
    expected_csv_hash = e155_status["artifact_sha256"][f"manifests/{panel}_SPLIT.csv"]
    expected_pkl_hash = e155_status["artifact_sha256"][f"manifests/{panel}_set2conditions.pkl"]
    if sha256_file(split_csv) != expected_csv_hash or sha256_file(split_pkl) != expected_pkl_hash:
        raise RuntimeError(f"{panel}: E155 split changed")
    with split_pkl.open("rb") as handle:
        split = pickle.load(handle)
    if {key: len(value) for key, value in split.items()} != {"train": 64, "val": 20, "test": 24}:
        raise RuntimeError(f"{panel}: unexpected split sizes")
    if set(split["train"]) & set(split["test"]) or set(split["val"]) & set(split["test"]):
        raise RuntimeError(f"{panel}: test task appears in development split")

    upstream_commit = subprocess.check_output(
        ["git", "-C", str(PRESCRIBE), "rev-parse", "HEAD"], text=True
    ).strip()
    if upstream_commit != EXPECTED_PRESCRIBE_COMMIT:
        raise RuntimeError(f"PRESCRIBE commit changed: {upstream_commit}")
    return {
        "e155_status": e155_status,
        "e156_status": e156_status,
        "panel_status": panel_status,
        "split": split,
        "split_csv": split_csv,
        "split_pkl": split_pkl,
        "dev_h5ad": dev_h5ad,
        "graph_path": graph_path,
        "test_expression_asset_present": False,
        "fixed_source_hashes": fixed_source_hashes,
        "e156_asset_hashes": e156_asset_hashes,
        "data_name": data_name,
        "task_vocabulary_audit": scgpt_task_vocabulary_audit(split),
    }


def load_development_pertdata(inputs: dict[str, object]):
    old_cwd = Path.cwd()
    old_path = list(sys.path)
    try:
        os.chdir(PRESCRIBE)
        sys.path.insert(0, str(PRESCRIBE))
        from src.data.pertdata import Get_Graph, LoadData  # noqa: PLC0415

        pert_data = LoadData(data_name=str(inputs["data_name"]), seed=SEED, backbone=None)
        edge_index, edge_weight, pert_reindex = Get_Graph(pert_data, SEED, overlap=True)
        if edge_index is not None or edge_weight is not None:
            raise RuntimeError("Unexpected graph edge tensors in native PRESCRIBE loader")
        if pert_reindex is None:
            raise RuntimeError("Missing perturbation reindex")
    finally:
        sys.path[:] = old_path
        os.chdir(old_cwd)

    split = inputs["split"]
    expected_dev = set(split["train"]) | set(split["val"])
    graph_keys = set(pert_data.dataset_processed)
    if graph_keys != expected_dev:
        raise RuntimeError(
            f"Development graph-key mismatch: missing={sorted(expected_dev-graph_keys)}, "
            f"extra={sorted(graph_keys-expected_dev)}"
        )
    if graph_keys & set(split["test"]):
        raise RuntimeError("Test task graph exists in E157 development loader")
    if set(pert_data.set2conditions["test"]) != set(split["test"]):
        raise RuntimeError("Frozen test task strings changed in loader")
    return pert_data, pert_reindex


def flatten_graphs(pert_data, conditions: list[str]) -> list[Any]:
    graphs: list[Any] = []
    for condition in conditions:
        items = pert_data.dataset_processed.get(condition)
        if not items:
            raise RuntimeError(f"No development graphs for {condition}")
        graphs.extend(items)
    return graphs


def structural_query_audit(
    pert_data: Any,
    split: dict[str, list[str]],
    dev_h5ad: Path,
) -> pd.DataFrame:
    import scanpy as sc

    adata = sc.read_h5ad(dev_h5ad, backed="r")
    try:
        ctrl_mask = adata.obs["condition"].astype(str).eq("ctrl").to_numpy()
        control = np.asarray(adata[ctrl_mask].to_memory().X.mean(axis=0)).reshape(-1).astype(np.float32)
        n_genes = int(adata.n_vars)
        gene_order_sha256 = sha256_text("\n".join(adata.var_names.astype(str)))
    finally:
        adata.file.close()
    labels = sorted(split["val"], key=lambda value: sha256_text(f"E157-dry|{value}"))[
        :N_EQUIVALENCE_TASKS
    ]
    rows = []
    for label in labels:
        graph = pert_data.dataset_processed[label][0]
        graph_x = graph.x.detach().cpu().numpy().reshape(-1).astype(np.float32)
        if len(graph_x) != n_genes:
            raise RuntimeError(f"{label}: graph gene dimension mismatch")
        rows.append(
            {
                "condition": label,
                "graph_pert": str(graph.pert),
                "gene_order_sha256": gene_order_sha256,
                "n_genes": n_genes,
                "max_abs_graph_x_minus_train_control": float(np.max(np.abs(graph_x - control))),
                "graph_has_y": "y" in set(graph.keys()),
                "graph_has_y_pca": "y_pca" in set(graph.keys()),
                "label_only_query_will_have_y": False,
                "label_only_query_will_have_y_pca": False,
            }
        )
    frame = pd.DataFrame(rows)
    if not np.allclose(
        frame["max_abs_graph_x_minus_train_control"], 0.0, atol=GRAPH_X_ATOL
    ):
        raise RuntimeError("Development native graph x differs from recomputed train control")
    return frame


def development_graph_audit(
    pert_data: Any,
    split: dict[str, list[str]],
    dev_h5ad: Path,
) -> pd.DataFrame:
    """Exhaustively verify every train/validation graph used by the optimizer."""
    import scanpy as sc
    import torch

    adata = sc.read_h5ad(dev_h5ad, backed="r")
    try:
        ctrl_mask = adata.obs["condition"].astype(str).eq("ctrl").to_numpy()
        control = torch.from_numpy(
            np.asarray(adata[ctrl_mask].to_memory().X.mean(axis=0))
            .reshape(-1)
            .astype(np.float32)
        )
        n_genes = int(adata.n_vars)
        expected_graph_counts = (
            adata.obs["condition"].astype(str).value_counts().to_dict()
        )
    finally:
        adata.file.close()

    role_by_condition = {
        condition: role
        for role in ["train", "val"]
        for condition in split[role]
    }
    observed_h5ad_conditions = set(expected_graph_counts)
    expected_development_conditions = set(role_by_condition)
    if observed_h5ad_conditions != expected_development_conditions:
        raise RuntimeError(
            "Development H5AD condition set mismatch: "
            f"missing={sorted(expected_development_conditions-observed_h5ad_conditions)}, "
            f"extra={sorted(observed_h5ad_conditions-expected_development_conditions)}"
        )
    if observed_h5ad_conditions & set(split["test"]):
        raise RuntimeError("Test condition appears in development H5AD")
    rows: list[dict[str, object]] = []
    for condition in sorted(role_by_condition):
        role = role_by_condition[condition]
        graphs = pert_data.dataset_processed.get(condition)
        if not graphs:
            raise RuntimeError(f"{condition}: no development graphs")
        max_x_delta = 0.0
        n_bad_x_shape = 0
        n_nonfinite_x = 0
        n_bad_pca_shape = 0
        n_nonfinite_pca = 0
        n_bad_rank_label = 0
        n_bad_condition_label = 0
        n_bad_pert_idx = 0
        for graph in graphs:
            flattened_x = graph.x.detach().cpu().reshape(-1)
            if flattened_x.numel() != n_genes:
                n_bad_x_shape += 1
            elif not bool(torch.isfinite(flattened_x).all()):
                n_nonfinite_x += 1
            else:
                max_x_delta = max(
                    max_x_delta,
                    float(torch.max(torch.abs(flattened_x - control)).item()),
                )
            y_pca = graph.y_pca.detach().cpu().reshape(-1)
            if y_pca.numel() != N_PCA:
                n_bad_pca_shape += 1
            elif not bool(torch.isfinite(y_pca).all()):
                n_nonfinite_pca += 1
            y_n = graph.y_n.detach().cpu().reshape(-1)
            rank_is_valid = (
                bool(torch.isfinite(y_n).all())
                if role == "train"
                else bool(torch.isnan(y_n).all())
            )
            n_bad_rank_label += int(not rank_is_valid)
            n_bad_condition_label += int(str(graph.pert) != condition)
            observed_pert_idx = np.asarray(graph.pert_idx, dtype=int).reshape(-1).tolist()
            expected_n_perturbations = len(
                [gene for gene in condition.split("+") if gene != "ctrl"]
            )
            if condition == "ctrl":
                pert_idx_is_valid = observed_pert_idx == [-1]
            else:
                pert_idx_is_valid = (
                    len(observed_pert_idx) == expected_n_perturbations
                    and all(value >= 0 for value in observed_pert_idx)
                    and -1 not in observed_pert_idx
                )
            n_bad_pert_idx += int(not pert_idx_is_valid)
        row = {
            "condition": condition,
            "split": role,
            "n_graphs": len(graphs),
            "n_graphs_expected_from_dev_h5ad": int(expected_graph_counts[condition]),
            "graph_count_delta": int(len(graphs) - expected_graph_counts[condition]),
            "n_genes": n_genes,
            "max_abs_x_minus_train_control": max_x_delta,
            "n_bad_x_shape": n_bad_x_shape,
            "n_nonfinite_x": n_nonfinite_x,
            "n_bad_pca10_shape": n_bad_pca_shape,
            "n_nonfinite_pca10": n_nonfinite_pca,
            "n_bad_rank_label": n_bad_rank_label,
            "rank_label_rule": "finite_train" if role == "train" else "nan_validation_sentinel",
            "n_bad_condition_label": n_bad_condition_label,
            "n_bad_pert_idx": n_bad_pert_idx,
        }
        rows.append(row)

    frame = pd.DataFrame(rows)
    count_lookup = frame.groupby("split")["condition"].nunique().to_dict()
    if count_lookup != {"train": 64, "val": 20}:
        raise RuntimeError(f"Development condition audit mismatch: {count_lookup}")
    failure_columns = [
        "n_bad_x_shape",
        "n_nonfinite_x",
        "n_bad_pca10_shape",
        "n_nonfinite_pca10",
        "n_bad_rank_label",
        "n_bad_condition_label",
        "n_bad_pert_idx",
        "graph_count_delta",
    ]
    if int(np.abs(frame[failure_columns].to_numpy(int)).sum()) != 0:
        raise RuntimeError("Malformed development graph detected")
    if float(frame["max_abs_x_minus_train_control"].max()) > GRAPH_X_ATOL:
        raise RuntimeError("A development graph does not use the frozen train-control mean")
    return frame


def scgpt_task_vocabulary_audit(split: dict[str, list[str]]) -> dict[str, object]:
    with FROZEN_E155_SOURCES["scGPT_perturbation_embedding"].open("rb") as handle:
        embedding = pickle.load(handle)
    all_conditions = [
        condition
        for role in ["train", "val", "test"]
        for condition in split[role]
    ]
    required = sorted(
        {
            gene
            for condition in all_conditions
            for gene in condition.split("+")
            if gene != "ctrl"
        }
    )
    missing = sorted(set(required) - set(embedding))
    if missing:
        raise RuntimeError(f"Frozen scGPT embedding lacks task genes: {missing}")
    return {
        "n_required_task_genes": len(required),
        "n_embedding_entries": len(embedding),
        "missing_task_genes": missing,
        "all_train_val_test_task_genes_encodable": True,
    }


def make_dev_datamodule(pert_data, split: dict[str, list[str]], seed: int):
    import lightning.pytorch as L
    import torch
    from torch_geometric.loader import DataLoader

    train_graphs = flatten_graphs(pert_data, split["train"])
    val_graphs = flatten_graphs(pert_data, split["val"])
    generator = torch.Generator()
    generator.manual_seed(seed)

    class DevelopmentOnlyDataModule(L.LightningDataModule):
        def __init__(self) -> None:
            super().__init__()
            self._train = DataLoader(
                train_graphs,
                batch_size=BATCH_SIZE,
                shuffle=True,
                drop_last=True,
                num_workers=0,
                generator=generator,
            )
            self._val = DataLoader(
                val_graphs,
                batch_size=BATCH_SIZE,
                shuffle=True,
                drop_last=False,
                num_workers=0,
                generator=generator,
            )

        def train_dataloader(self):
            return self._train

        def val_dataloader(self):
            return self._val

        def test_dataloader(self):
            raise RuntimeError("E157 has no test dataloader or test-expression asset")

    return DevelopmentOnlyDataModule(), len(train_graphs), len(val_graphs)


def native_args(data_name: str, seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        seed=seed,
        data_name=data_name,
        backbone=None,
        batch_size=BATCH_SIZE,
        latent_dim=64,
        output_dim=10,
        flow_layers=10,
        flow_size=0.774,
        flow_n_hidden=2,
        maf_layers=10,
        budget="exp",
        bound=30,
        warmup_epochs=N_WARMUP_EPOCHS,
        warmup_lr=1e-3,
        log_prob_positive=False,
        accumulate_grad_batches=4,
        load_from="",
        lr=1e-4,
        lam1=1e-7,
        scheduler="plateau",
        interval="epoch",
        change_step=2,
        reduce_rate=0.99,
        warmup_steps=0,
        warmup_max_steps=int(875 * 1.25),
        lam2=0.1,
        lam3=1e-5,
    )


def clean_hparams(module: Any) -> None:
    for key in ["adata", "model"]:
        if key in module.hparams:
            module.hparams.pop(key)
    for key, value in module.hparams.items():
        if value.__class__.__module__.startswith("anndata"):
            raise RuntimeError(f"AnnData remains in checkpoint hyperparameters: {key}")


def forbidden_checkpoint_objects(value: Any, path: str = "root") -> list[str]:
    import torch

    issues: list[str] = []
    module = value.__class__.__module__
    if module.startswith("anndata"):
        return [f"{path}:AnnData"]
    if isinstance(value, str) and "sealed_test_transform" in value:
        return [f"{path}:legacy-test-expression-path"]
    if isinstance(value, dict):
        for key, item in value.items():
            issues.extend(forbidden_checkpoint_objects(item, f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            issues.extend(forbidden_checkpoint_objects(item, f"{path}[{idx}]"))
    elif isinstance(value, torch.Tensor):
        return issues
    return issues


def audit_checkpoint(path: Path) -> dict[str, object]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    issues = forbidden_checkpoint_objects(payload)
    hparams = payload.get("hyper_parameters", {}) if isinstance(payload, dict) else {}
    if issues:
        raise RuntimeError(f"Checkpoint contains forbidden objects: {issues[:10]}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "contains_anndata": False,
        "contains_test_expression_path": False,
        "hyperparameter_keys": sorted(map(str, hparams.keys())),
    }


def save_slim_checkpoint(module: Any, path: Path, metadata: dict[str, object]) -> dict[str, object]:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite locked E157 checkpoint: {path}")
    state = {key: tensor.detach().cpu() for key, tensor in module.state_dict().items()}
    payload = {
        "schema_version": "safeconf_e157_locked_native_state_v1",
        "state_dict": state,
        "metadata": metadata,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return audit_checkpoint(path)


def train_native(
    config: dict[str, object],
    inputs: dict[str, object],
    pert_data: Any,
    datamodule: Any,
    run_dir: Path,
    status_path: Path,
) -> tuple[Any, dict[str, object]]:
    import lightning.pytorch as L
    import torch
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger

    old_cwd = Path.cwd()
    old_path = list(sys.path)
    try:
        os.chdir(PRESCRIBE)
        sys.path.insert(0, str(PRESCRIBE))
        from Step2_train import build_model  # noqa: PLC0415
        from src.model import (  # noqa: PLC0415
            NaturalPosteriorNetworkFlowLightningModule,
            NaturalPosteriorNetworkLightningModule,
        )

        seed_everything(SEED, workers=True)
        args = native_args(str(config["data_name"]), SEED)
        model = build_model(args, pert_data)
        warmup_weights = run_dir / "checkpoints/warmup_complete_model_state.pt"
        warmup_last = run_dir / "checkpoints/warmup/last.ckpt"
        warmup_weights.parent.mkdir(parents=True, exist_ok=True)
        if warmup_weights.exists():
            current_status = json.loads(status_path.read_text(encoding="utf-8"))
            recorded_warmup_hash = current_status.get("warmup_state_sha256")
            if not recorded_warmup_hash or sha256_file(warmup_weights) != recorded_warmup_hash:
                raise RuntimeError("Existing warmup state lacks a matching recorded SHA256")
            model.load_state_dict(torch.load(warmup_weights, map_location="cpu", weights_only=True))
        else:
            update_status(
                status_path,
                phase="warmup_started",
                warmup_epochs=N_WARMUP_EPOCHS,
                model_training_started=True,
                test_expression_accessed=False,
            )
            warmup = NaturalPosteriorNetworkFlowLightningModule(
                model,
                learning_rate=args.warmup_lr,
                early_stopping=False,
                stage="warmup",
                log_prob_positive=args.log_prob_positive,
            )
            clean_hparams(warmup)
            callback = ModelCheckpoint(
                dirpath=str(warmup_last.parent),
                save_last=True,
                save_top_k=0,
                every_n_epochs=1,
            )
            trainer = Trainer(
                deterministic=True,
                callbacks=[callback],
                max_epochs=N_WARMUP_EPOCHS,
                devices=1,
                accelerator="gpu",
                logger=TensorBoardLogger(str(run_dir / "logs"), name="warmup"),
                log_every_n_steps=1,
                check_val_every_n_epoch=1,
                accumulate_grad_batches=args.accumulate_grad_batches,
            )
            resume = None
            if warmup_last.exists():
                current_status = json.loads(status_path.read_text(encoding="utf-8"))
                prior_phase = str(current_status.get("prior_phase_before_current_invocation", ""))
                if prior_phase not in {
                    "warmup_started",
                    "failed_no_test_truth_access_requires_audit",
                }:
                    raise RuntimeError(
                        f"Orphan warmup checkpoint is not eligible for recovery: {prior_phase}"
                    )
                warmup_last_audit = audit_checkpoint(warmup_last)
                update_status(
                    status_path,
                    resumed_from_warmup_last=True,
                    resumed_warmup_last_sha256=warmup_last_audit["sha256"],
                )
                resume = str(warmup_last)
            trainer.fit(model=warmup, datamodule=datamodule, ckpt_path=resume)
            torch.save(model.state_dict(), warmup_weights)
            update_status(
                status_path,
                phase="warmup_complete",
                warmup_state_sha256=sha256_file(warmup_weights),
            )

        module = NaturalPosteriorNetworkLightningModule(
            model=model,
            learning_rate_decay=True,
            learning_rate=args.lr,
            lam1=args.lam1,
            scheduler=args.scheduler,
            interval=args.interval,
            change_step=args.change_step,
            reduce_rate=args.reduce_rate,
            patience=3,
            warmup_steps=args.warmup_steps,
            warmup_max_steps=args.warmup_max_steps,
            lam3=args.lam3,
            lam2=args.lam2,
            save_value_only=False,
            data_name=args.data_name,
            adata=None,
        )
        clean_hparams(module)
        main_dir = run_dir / "checkpoints/main"
        checkpoint = ModelCheckpoint(
            dirpath=str(main_dir),
            filename="best-{epoch:02d}",
            auto_insert_metric_name=False,
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            every_n_epochs=1,
        )
        last_path = main_dir / "last.ckpt"
        update_status(
            status_path,
            phase="native_training_started",
            max_epochs=N_EPOCHS,
            model_training_started=True,
            test_expression_accessed=False,
        )
        trainer = Trainer(
            deterministic=True,
            callbacks=[checkpoint],
            max_epochs=N_EPOCHS,
            devices=1,
            accelerator="gpu",
            logger=TensorBoardLogger(str(run_dir / "logs"), name="native"),
            log_every_n_steps=1,
            check_val_every_n_epoch=1,
            accumulate_grad_batches=args.accumulate_grad_batches,
        )
        resume = None
        if last_path.exists():
            current_status = json.loads(status_path.read_text(encoding="utf-8"))
            prior_phase = str(current_status.get("prior_phase_before_current_invocation", ""))
            if prior_phase not in {
                "native_training_started",
                "failed_no_test_truth_access_requires_audit",
            }:
                raise RuntimeError(
                    f"Orphan main checkpoint is not eligible for recovery: {prior_phase}"
                )
            last_audit = audit_checkpoint(last_path)
            update_status(
                status_path,
                resumed_from_main_last=True,
                resumed_main_last_sha256=last_audit["sha256"],
            )
            resume = str(last_path)
        trainer.fit(model=module, datamodule=datamodule, ckpt_path=resume)
        if not checkpoint.best_model_path:
            raise RuntimeError("No best checkpoint was selected from finite val/loss")
        if checkpoint.best_model_score is None or not bool(
            torch.isfinite(checkpoint.best_model_score).all()
        ):
            raise RuntimeError("Best validation loss is missing or non-finite")
        best_path = Path(checkpoint.best_model_path)
        if not best_path.is_file():
            raise FileNotFoundError("Selected best native checkpoint does not exist")
        best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
        module.load_state_dict(best_payload["state_dict"])
        lightning_audit = audit_checkpoint(best_path)
        slim_path = (
            LOCKED_MODEL_ROOT
            / str(config["data_name"])
            / "E157_LOCKED_NATIVE_STATE.pt"
        )
        slim_audit = save_slim_checkpoint(
            module,
            slim_path,
            {
                "panel": config["panel"],
                "data_name": config["data_name"],
                "seed": SEED,
                "max_epochs": N_EPOCHS,
                "warmup_epochs": N_WARMUP_EPOCHS,
                "batch_size": BATCH_SIZE,
                "source_manifest_sha256": json.loads(
                    status_path.read_text(encoding="utf-8")
                )["source_manifest_sha256"],
                "input_manifest_sha256": json.loads(
                    status_path.read_text(encoding="utf-8")
                )["input_manifest_sha256"],
                "runner_sha256": sha256_file(Path(__file__).resolve()),
                "prescribe_commit": EXPECTED_PRESCRIBE_COMMIT,
                "e156_dev_h5ad_sha256": inputs["e156_status"]["perturb_processed_sha256"][config["panel"]],
                "e156_cell_graphs_sha256": inputs["e156_status"]["cell_graphs_sha256"][config["panel"]],
            },
        )
        locked = {
            "best_lightning_checkpoint": lightning_audit,
            "locked_slim_checkpoint": slim_audit,
            "best_val_loss": float(checkpoint.best_model_score.cpu()),
        }
        update_status(
            status_path,
            phase="checkpoint_locked_before_test_task_query",
            checkpoint_locked_at=datetime.now().isoformat(timespec="seconds"),
            checkpoint_audit=locked,
            test_expression_accessed=False,
        )
        return module, locked
    finally:
        sys.path[:] = old_path
        os.chdir(old_cwd)


def control_and_pca(
    dev_h5ad: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    import scanpy as sc

    adata = sc.read_h5ad(dev_h5ad, backed="r")
    try:
        ctrl_mask = adata.obs["condition"].astype(str).eq("ctrl").to_numpy()
        control = np.asarray(adata[ctrl_mask].to_memory().X.mean(axis=0)).reshape(-1).astype(np.float32)
        mean = np.asarray(adata.uns["pca_mean"], dtype=np.float32)
        components = np.asarray(adata.uns["pca_components"], dtype=np.float32)
        genes = adata.var_names.astype(str).tolist()
    finally:
        adata.file.close()
    return control, mean, components, genes


def label_only_batch(conditions: list[str], control: np.ndarray):
    import torch
    from torch_geometric.data import Batch, Data

    graphs = [
        Data(x=torch.from_numpy(control.copy()).float().unsqueeze(1), pert=condition)
        for condition in conditions
    ]
    batch = Batch.from_data_list(graphs)
    stored_truth_keys = [key for key in ["y", "y_pca"] if key in set(batch.keys())]
    nonnull_truth_attributes = [
        key for key in ["y", "y_pca"] if getattr(batch, key, None) is not None
    ]
    if stored_truth_keys or nonnull_truth_attributes:
        raise RuntimeError("Label-only query unexpectedly contains truth")
    return batch


def posterior_fields(module: Any, batch: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import torch

    with torch.inference_mode():
        posterior, log_prob = module.model(batch)
        pca_prediction = posterior.maximum_a_posteriori().loc
        epistemic, aleatoric = module._calculate_unc(posterior, log_prob=log_prob)
    return (
        pca_prediction.detach().cpu().numpy(),
        log_prob.detach().cpu().numpy(),
        epistemic.detach().cpu().numpy(),
        aleatoric.detach().cpu().numpy(),
    )


def forward_equivalence_audit(
    module: Any,
    pert_data: Any,
    split: dict[str, list[str]],
    control: np.ndarray,
    device: Any,
) -> pd.DataFrame:
    from torch_geometric.data import Batch

    conditions = sorted(split["val"], key=lambda value: sha256_text(f"E157-forward|{value}"))[
        :N_EQUIVALENCE_TASKS
    ]
    native = Batch.from_data_list([pert_data.dataset_processed[value][0] for value in conditions]).to(device)
    query = label_only_batch(conditions, control).to(device)
    native_fields = posterior_fields(module, native)
    query_fields = posterior_fields(module, query)
    rows = []
    for index, condition in enumerate(conditions):
        row = {"condition": condition}
        names = ["pca_prediction", "log_prob", "epistemic", "aleatoric"]
        for name, native_value, query_value in zip(names, native_fields, query_fields):
            native_array = np.asarray(native_value[index])
            query_array = np.asarray(query_value[index])
            if not np.isfinite(native_array).all() or not np.isfinite(query_array).all():
                raise RuntimeError(f"Non-finite {name} in development forward equivalence")
            delta = np.abs(native_array - query_array)
            if not np.isfinite(delta).all():
                raise RuntimeError(f"Non-finite {name} delta in development forward equivalence")
            row[f"max_abs_delta_{name}"] = float(np.max(delta))
        rows.append(row)
    frame = pd.DataFrame(rows)
    delta_columns = [column for column in frame if column.startswith("max_abs_delta_")]
    if frame[delta_columns].to_numpy(float).max() > FORWARD_EQUIVALENCE_ATOL:
        raise RuntimeError("Label-only forward is not equivalent to native development graph forward")
    return frame


def lock_test_task_scores(
    module: Any,
    dev_h5ad: Path,
    split: dict[str, list[str]],
    run_dir: Path,
    panel: str,
) -> dict[str, object]:
    import torch

    control, pca_mean, components, genes = control_and_pca(dev_h5ad)
    device = next(module.parameters()).device
    tasks = list(split["test"])
    batch = label_only_batch(tasks, control).to(device)
    pca_prediction, log_prob, epistemic, aleatoric = posterior_fields(module, batch)
    reconstructed = pca_prediction @ components + pca_mean
    effect = reconstructed - control[None, :]
    magnitude = np.sqrt(np.mean(effect**2, axis=1))
    combined = 2.0 * epistemic + aleatoric
    rows = []
    for index, task in enumerate(tasks):
        row = {
            "panel": panel,
            "task_id": task,
            "query_has_test_expression": False,
            "query_has_y": False,
            "query_has_y_pca": False,
            "log_prob": float(np.asarray(log_prob[index]).reshape(-1)[0]),
            "epistemic_confidence": float(np.asarray(epistemic[index]).reshape(-1)[0]),
            "aleatoric_confidence": float(np.asarray(aleatoric[index]).reshape(-1)[0]),
            "combined_confidence_official": float(np.asarray(combined[index]).reshape(-1)[0]),
            "predicted_magnitude_rms": float(magnitude[index]),
            "gene_order_sha256": sha256_text("\n".join(genes)),
        }
        row.update(
            {
                f"predicted_pca_{dim}": float(pca_prediction[index, dim])
                for dim in range(N_PCA)
            }
        )
        rows.append(row)
    table = pd.DataFrame(rows)
    if len(table) != 24 or table["task_id"].duplicated().any():
        raise RuntimeError("Locked test task-score table does not contain 24 unique tasks")
    if set(table["task_id"]) != set(split["test"]):
        raise RuntimeError("Locked task-score identifiers differ from the E155 test contract")
    numeric = table.select_dtypes(include=[np.number]).to_numpy(float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("Non-finite locked test task score")
    path = run_dir / "locked/E157_LOCKED_LABEL_ONLY_TASK_SCORES.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    table.to_csv(temporary, index=False)
    temporary.replace(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "n_tasks": len(table),
        "contains_test_truth": False,
        "contains_test_expression": False,
        "fields": table.columns.tolist(),
    }


def run_dry(config: dict[str, object], inputs: dict[str, object], run_dir: Path, status_path: Path) -> None:
    pert_data, pert_reindex = load_development_pertdata(inputs)
    structure = structural_query_audit(
        pert_data, inputs["split"], Path(inputs["dev_h5ad"])
    )
    structure.to_csv(run_dir / "LOADER_DRYRUN_AUDIT.csv", index=False)
    graph_audit = development_graph_audit(
        pert_data, inputs["split"], Path(inputs["dev_h5ad"])
    )
    graph_audit_path = run_dir / "DEVELOPMENT_GRAPH_AUDIT.csv"
    graph_audit.to_csv(graph_audit_path, index=False)
    control, _, _, _ = control_and_pca(Path(inputs["dev_h5ad"]))
    query_conditions = sorted(
        inputs["split"]["val"], key=lambda value: sha256_text(f"E157-query-dry|{value}")
    )[:N_EQUIVALENCE_TASKS]
    label_query = label_only_batch(query_conditions, control)
    observed_query_conditions = [str(value) for value in label_query.pert]
    if observed_query_conditions != query_conditions:
        raise RuntimeError("Label-only dry-run perturbation order changed during batching")
    label_query_audit = {
        "n_queries": int(label_query.num_graphs),
        "conditions": query_conditions,
        "observed_batched_conditions": observed_query_conditions,
        "stored_keys": sorted(label_query.keys()),
        "y_key_stored": "y" in set(label_query.keys()),
        "y_pca_key_stored": "y_pca" in set(label_query.keys()),
        "y_attribute_nonnull": getattr(label_query, "y", None) is not None,
        "y_pca_attribute_nonnull": getattr(label_query, "y_pca", None) is not None,
        "x_rows": int(label_query.x.shape[0]),
        "x_columns": int(label_query.x.shape[1]),
    }
    if label_query_audit["n_queries"] != N_EQUIVALENCE_TASKS:
        raise RuntimeError("Label-only dry-run query count mismatch")
    if label_query_audit["x_rows"] != N_EQUIVALENCE_TASKS * len(control):
        raise RuntimeError("Label-only dry-run x shape mismatch")
    atomic_json(run_dir / "LABEL_ONLY_BATCH_DRYRUN.json", label_query_audit)
    datamodule, n_train, n_val = make_dev_datamodule(pert_data, inputs["split"], SEED)
    train_batch = next(iter(datamodule.train_dataloader()))
    val_batch = next(iter(datamodule.val_dataloader()))
    if train_batch.num_graphs > BATCH_SIZE or val_batch.num_graphs > BATCH_SIZE:
        raise RuntimeError("Dry-run batch size exceeded contract")
    update_status(
        status_path,
        phase="dry_run_complete_no_model_built_no_training_no_test_truth_access",
        finished_at=datetime.now().isoformat(timespec="seconds"),
        n_train_graphs=n_train,
        n_val_graphs=n_val,
        n_test_graphs=0,
        n_train_batch_graphs=int(train_batch.num_graphs),
        n_val_batch_graphs=int(val_batch.num_graphs),
        n_nodes=int(pert_data.nodes_num),
        n_perturbations=int(pert_data.num_pert),
        perturb_reindex_size=len(pert_reindex),
        structural_query_audit_sha256=sha256_file(run_dir / "LOADER_DRYRUN_AUDIT.csv"),
        development_graph_audit_sha256=sha256_file(graph_audit_path),
        development_graph_audit_n_conditions=int(len(graph_audit)),
        development_graph_audit_n_graphs=int(graph_audit["n_graphs"].sum()),
        development_graph_max_abs_x_delta=float(
            graph_audit["max_abs_x_minus_train_control"].max()
        ),
        label_only_batch_dryrun_sha256=sha256_file(
            run_dir / "LABEL_ONLY_BATCH_DRYRUN.json"
        ),
        label_only_batch_dryrun=label_query_audit,
        model_built=False,
        model_training_started=False,
        test_task_scores_generated=False,
        test_expression_accessed=False,
    )


def run_formal(config: dict[str, object], inputs: dict[str, object], run_dir: Path, status_path: Path) -> None:
    import torch
    from lightning.pytorch import seed_everything

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Formal E157 requires exactly one visible CUDA GPU per process")
    locked_model_path = (
        LOCKED_MODEL_ROOT
        / str(config["data_name"])
        / "E157_LOCKED_NATIVE_STATE.pt"
    )
    if locked_model_path.exists():
        raise FileExistsError(
            f"Locked model asset already exists and requires manual audit: {locked_model_path}"
        )
    seed_everything(SEED, workers=True)
    pert_data, _ = load_development_pertdata(inputs)
    structural = structural_query_audit(
        pert_data, inputs["split"], Path(inputs["dev_h5ad"])
    )
    structural.to_csv(run_dir / "LOADER_STRUCTURAL_AUDIT.csv", index=False)
    graph_audit = development_graph_audit(
        pert_data, inputs["split"], Path(inputs["dev_h5ad"])
    )
    graph_audit_path = run_dir / "DEVELOPMENT_GRAPH_AUDIT.csv"
    graph_audit.to_csv(graph_audit_path, index=False)
    datamodule, n_train, n_val = make_dev_datamodule(pert_data, inputs["split"], SEED)
    update_status(
        status_path,
        phase="formal_preflight_complete",
        n_train_graphs=n_train,
        n_val_graphs=n_val,
        n_test_graphs=0,
        visible_gpu=torch.cuda.get_device_name(0),
        locked_model_path=str(locked_model_path),
        development_graph_audit_sha256=sha256_file(graph_audit_path),
        development_graph_audit_n_conditions=int(len(graph_audit)),
        development_graph_audit_n_graphs=int(graph_audit["n_graphs"].sum()),
        development_graph_max_abs_x_delta=float(
            graph_audit["max_abs_x_minus_train_control"].max()
        ),
        test_expression_accessed=False,
    )
    module, checkpoint_audit = train_native(
        config, inputs, pert_data, datamodule, run_dir, status_path
    )
    device = torch.device("cuda:0")
    module.to(device).eval()
    control, _, _, _ = control_and_pca(Path(inputs["dev_h5ad"]))
    equivalence = forward_equivalence_audit(
        module, pert_data, inputs["split"], control, device
    )
    equivalence_path = run_dir / "E157_DEVELOPMENT_FORWARD_EQUIVALENCE.csv"
    equivalence.to_csv(equivalence_path, index=False)
    score_audit = lock_test_task_scores(
        module,
        Path(inputs["dev_h5ad"]),
        inputs["split"],
        run_dir,
        str(config["panel"]),
    )
    update_status(
        status_path,
        phase="complete_checkpoint_and_label_only_scores_locked_no_test_truth_access",
        finished_at=datetime.now().isoformat(timespec="seconds"),
        checkpoint_audit=checkpoint_audit,
        development_forward_equivalence_sha256=sha256_file(equivalence_path),
        development_forward_max_abs_delta=float(
            equivalence[[column for column in equivalence if column.startswith("max_abs_delta_")]]
            .to_numpy(float)
            .max()
        ),
        locked_task_score_audit=score_audit,
        test_expression_accessed=False,
        test_endpoint_computed=False,
    )


def main() -> None:
    cli = parse_args()
    if cli.seed != SEED:
        raise RuntimeError("E157 contract fixes seed=3407")
    config = dict(PANEL_CONFIG[cli.panel])
    physical_gpu = int(config["physical_gpu"] if cli.gpu_index is None else cli.gpu_index)
    if physical_gpu != int(config["physical_gpu"]):
        raise RuntimeError(f"{cli.panel} is frozen to physical GPU {config['physical_gpu']}")
    os.environ.update(
        {
            "CUDA_VISIBLE_DEVICES": str(physical_gpu),
            "CUDA_LAUNCH_BLOCKING": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "WANDB_MODE": "offline",
        }
    )
    run_name = f"{config['data_name']}_{cli.mode.replace('-', '')}_seed{SEED}"
    run_dir = OUT / run_name
    status_path = run_dir / "STATUS.json"
    if cli.mode == "dry-run" and run_dir.exists():
        raise FileExistsError(f"Dry-run output already exists: {run_dir}")
    prior: dict[str, object] = {}
    if cli.mode == "formal" and status_path.exists():
        prior = json.loads(status_path.read_text(encoding="utf-8"))
        if str(prior.get("phase", "")).startswith("complete_"):
            raise RuntimeError("Formal E157 run already complete; refusing to overwrite")
    if (
        cli.mode == "formal"
        and run_dir.exists()
        and not status_path.exists()
        and any(run_dir.iterdir())
    ):
        raise RuntimeError("Formal run directory has artifacts but no STATUS; refusing orphan recovery")
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    git_provenance = formal_git_provenance() if cli.mode == "formal" else {}
    update_status(
        status_path,
        experiment="E157_prescribe_norman_p3p4_native",
        panel=config["panel"],
        data_name=config["data_name"],
        mode=cli.mode,
        seed=SEED,
        physical_gpu=physical_gpu,
        phase="started_input_validation",
        started_at=datetime.now().isoformat(timespec="seconds"),
        native_epochs=N_EPOCHS,
        warmup_epochs=N_WARMUP_EPOCHS,
        batch_size=BATCH_SIZE,
        model_training_started=False,
        test_expression_accessed=False,
        test_endpoint_computed=False,
        interrupted_resume_bitwise_identity_claimed=False,
        prior_phase_before_current_invocation=prior.get("phase"),
        **git_provenance,
    )
    try:
        if not CONTRACT.exists():
            raise FileNotFoundError(CONTRACT)
        inputs = validate_inputs(config)
        sources = source_manifest()
        source_manifest_sha256 = write_frozen_csv(run_dir / "SOURCE_MANIFEST.csv", sources)
        frozen_inputs = input_manifest(inputs)
        input_manifest_sha256 = write_frozen_csv(run_dir / "INPUT_MANIFEST.csv", frozen_inputs)
        update_status(
            status_path,
            phase="inputs_verified",
            contract_path=str(CONTRACT),
            contract_sha256=sha256_file(CONTRACT),
            source_manifest_sha256=source_manifest_sha256,
            input_manifest_sha256=input_manifest_sha256,
            e156_dev_h5ad_sha256=inputs["e156_status"]["perturb_processed_sha256"][config["panel"]],
            e156_cell_graphs_sha256=inputs["e156_status"]["cell_graphs_sha256"][config["panel"]],
            e155_split_csv_sha256=sha256_file(inputs["split_csv"]),
            e155_split_pkl_sha256=sha256_file(inputs["split_pkl"]),
            test_expression_asset_present=inputs["test_expression_asset_present"],
            task_vocabulary_audit=inputs["task_vocabulary_audit"],
            test_expression_accessed=False,
        )
        if cli.mode == "dry-run":
            run_dry(config, inputs, run_dir, status_path)
        else:
            run_formal(config, inputs, run_dir, status_path)
        update_status(status_path, runtime_seconds=round(time.time() - started, 3))
        print(status_path.read_text(encoding="utf-8"))
    except Exception as exc:
        update_status(
            status_path,
            phase="failed_no_test_truth_access_requires_audit",
            failed_at=datetime.now().isoformat(timespec="seconds"),
            runtime_seconds=round(time.time() - started, 3),
            error=repr(exc),
            traceback=traceback.format_exc(),
            test_expression_accessed=False,
            test_endpoint_computed=False,
        )
        raise


if __name__ == "__main__":
    main()
