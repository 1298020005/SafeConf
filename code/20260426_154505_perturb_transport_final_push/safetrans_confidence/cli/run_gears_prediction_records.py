#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

LOCAL_ATLAS_FILES = {
    "norman": Path("/home/yyf/data/singlecell_perturbation_atlas/official_generalization/Norman.h5ad"),
    "adamson": Path("/home/yyf/data/singlecell_perturbation_atlas/official_generalization/Adamson.h5ad"),
    "dixit": Path("/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/DixitRegev2016_K562_TFs_7_days.h5ad"),
    "frangieh": Path("/home/yyf/data/singlecell_perturbation_atlas/official_generalization/Frangieh.h5ad"),
}


def _is_ctrl_like(values: pd.Series) -> pd.Series:
    return (
        values.str.lower().isin({"control", "ctrl", "non-targeting", "nt", "mock", "vehicle", "dmso"})
        | values.str.contains("control", case=False, na=False)
        | values.str.contains("ctrl", case=False, na=False)
        | values.eq("nan")
    )


def load_local_gears_adata(data_name: str, max_genes: int):
    import scanpy as sc
    import scipy.sparse as sp

    path = LOCAL_ATLAS_FILES[data_name]
    if not path.exists():
        raise FileNotFoundError(path)
    adata = sc.read_h5ad(path)
    if "perturbation" not in adata.obs.columns:
        raise ValueError(f"local atlas file missing perturbation column: {path}")

    gene_name_set = set(adata.var_names.astype(str))
    pert = adata.obs["perturbation"].astype(str).fillna("control")
    source = pert
    if "target" in adata.obs.columns:
        target = adata.obs["target"].astype(str).fillna("control")
        if int(target.isin(gene_name_set).sum()) > int(pert.isin(gene_name_set).sum()):
            keep = _is_ctrl_like(target) | target.isin(gene_name_set)
            if not bool(keep.all()):
                adata = adata[keep.values].copy()
                target = target.loc[adata.obs_names]
            source = target

    if max_genes and adata.n_vars > max_genes:
        required_genes = {
            gene
            for condition in source.astype(str).unique()
            for gene in str(condition).split("+")
            if gene not in {"ctrl", "nan"} and gene in gene_name_set
        }
        if "ncounts" in adata.var.columns:
            rank_values = pd.to_numeric(adata.var["ncounts"], errors="coerce").fillna(0)
        elif "ncells" in adata.var.columns:
            rank_values = pd.to_numeric(adata.var["ncells"], errors="coerce").fillna(0)
        else:
            rank_values = pd.Series(range(adata.n_vars, 0, -1), index=adata.var_names)
        keep_genes = set(required_genes)
        for gene in rank_values.sort_values(ascending=False).index.astype(str):
            if len(keep_genes) >= max_genes:
                break
            keep_genes.add(gene)
        adata = adata[:, adata.var_names.astype(str).isin(keep_genes)].copy()

    pert = source.astype(str).fillna("control").copy()
    ctrl_like = _is_ctrl_like(pert)
    single_like = (~ctrl_like) & (~pert.str.contains(r"\+", regex=True, na=False))
    pert.loc[single_like] = pert.loc[single_like] + "+ctrl"
    if "celltype" in adata.obs.columns:
        cell_type = adata.obs["celltype"].astype(str).fillna("cell")
    elif "cell_type" in adata.obs.columns:
        cell_type = adata.obs["cell_type"].astype(str).fillna("cell")
    elif "cell_line" in adata.obs.columns:
        cell_type = adata.obs["cell_line"].astype(str).fillna("cell")
    elif "condition" in adata.obs.columns:
        cell_type = adata.obs["condition"].astype(str).fillna("cell")
    else:
        cell_type = pd.Series(["cell"] * adata.n_obs, index=adata.obs_names)
    adata.obs = adata.obs.copy()
    adata.obs["condition"] = np.where(ctrl_like, "ctrl", pert)
    adata.obs["cell_type"] = cell_type.astype(str).values
    if "gene_name" not in adata.var.columns:
        adata.var["gene_name"] = adata.var_names.astype(str)
    adata.X = adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    return adata


def _dense_mean(x) -> np.ndarray:
    arr = x
    if hasattr(arr, "toarray"):
        arr = arr.toarray()
    return np.asarray(arr, dtype=np.float32).mean(axis=0)


def _gene_order_hash(gene_names: list[str]) -> str:
    payload = "\n".join(map(str, gene_names)).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _string_sequence_hash(values) -> str:
    payload = "\0".join(map(str, values)).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _set_all_random_seeds(seed: int) -> None:
    """Control every RNG used by the legacy GEARS stack.

    The upstream ``gears.gears`` module sets ``torch.manual_seed(0)`` at import
    time.  Calling this helper after import and again immediately before model
    initialisation makes the requested replicate seed real rather than a label.
    """
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _state_mapping_sha256(state_dict: dict) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _state_dict_sha256(model: torch.nn.Module) -> str:
    return _state_mapping_sha256(model.state_dict())


def _dataset_group(dataset: str) -> str:
    if dataset in {"norman", "adamson", "dixit", "frangieh"}:
        return "gears_crispr_group"
    return "unknown_group"


def _read_requested_test_conditions(path: Path, available_conditions: set[str]) -> list[str]:
    """Read a fixed GEARS test perturbation list and map gene names to conditions.

    Accepted inputs:
    - text file with one perturbation or condition per line;
    - CSV with a column named condition, perturbation, task_gene or task_id;
    - CSV without those columns, where the first column is used.

    GEARS single-gene conditions are usually stored as ``GENE+ctrl``.  For
    convenience a plain gene name in the input is mapped to ``GENE+ctrl`` when
    available.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        value_col = None
        for col in ["condition", "perturbation", "task_gene", "task_id"]:
            if col in df.columns:
                value_col = col
                break
        if value_col is None:
            value_col = df.columns[0]
        raw_values = df[value_col].dropna().astype(str).tolist()
    else:
        raw_values = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    requested: list[str] = []
    missing: list[str] = []
    for raw in raw_values:
        value = str(raw).strip()
        if not value:
            continue
        candidates = [value]
        if not value.endswith("+ctrl"):
            candidates.append(value + "+ctrl")
        if value.endswith("+ctrl"):
            candidates.append(value.replace("+ctrl", ""))
        match = next((c for c in candidates if c in available_conditions), None)
        if match is None:
            missing.append(value)
        elif match != "ctrl" and match not in requested:
            requested.append(match)
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"requested test perturbations not found in GEARS conditions: {preview}")
    if not requested:
        raise ValueError(f"no usable test perturbations found in {path}")
    return requested


def _override_fixed_test_conditions(
    pert_data,
    requested_test_conditions: list[str],
    deterministic_val: bool = False,
) -> dict:
    """Override GEARS set2conditions so the same tasks are tested across seeds."""
    available = [str(c) for c in pert_data.adata.obs["condition"].astype(str).unique()]
    requested = [c for c in requested_test_conditions if c in available and c != "ctrl"]
    requested_set = set(requested)
    all_non_ctrl = sorted(c for c in available if c != "ctrl")
    train_pool = [c for c in all_non_ctrl if c not in requested_set]
    if deterministic_val:
        n_val = max(1, min(len(train_pool) // 10 or 1, len(train_pool) - 1))
        val = train_pool[:n_val]
        train = train_pool[n_val:]
    else:
        old_val = [
            str(c)
            for c in pert_data.set2conditions.get("val", [])
            if str(c) in all_non_ctrl and str(c) not in requested_set
        ]
        val_set = set(old_val)
        train = [c for c in train_pool if c not in val_set]
        val = old_val
    if not val and len(train) >= 2:
        n_val = max(1, min(len(train) // 5 or 1, len(train) - 1))
        val = train[:n_val]
        train = train[n_val:]
    if not train:
        raise ValueError("fixed test split leaves no non-control train perturbations")
    train_with_ctrl = ["ctrl"] + train if "ctrl" in available else train
    pert_data.set2conditions = {
        "train": train_with_ctrl,
        "val": val,
        "test": requested,
    }
    split_map = {c: "train" for c in train_with_ctrl}
    split_map.update({c: "val" for c in val})
    split_map.update({c: "test" for c in requested})
    pert_data.adata.obs["split"] = pert_data.adata.obs["condition"].astype(str).map(split_map)
    return {
        "fixed_test_conditions": requested,
        "n_fixed_test_conditions": len(requested),
        "n_train_conditions_after_override": len(train_with_ctrl),
        "n_val_conditions_after_override": len(val),
        "fixed_test_deterministic_val": bool(deterministic_val),
    }


def _subsample_train_val_graphs(
    pert_data,
    max_cells_per_condition: int,
    sampling_seed: int,
) -> dict:
    """Deterministically bound training cost without touching test graphs.

    GEARS stores one graph per target cell in ``dataset_processed``.  For a
    fixed-task audit, retaining every training cell can turn a replication
    into hours of repeated, near-identical optimization while the held-out
    task panel remains small.  This helper samples within each *training or
    validation condition* before loaders are constructed.  Test condition
    lists are never shortened.  The selection uses graph positions and a
    declared seed only, never target effects, prediction errors, or metrics.
    """
    if max_cells_per_condition <= 0:
        return {
            "enabled": False,
            "max_cells_per_condition": int(max_cells_per_condition),
            "sampling_seed": int(sampling_seed),
            "conditions": {},
        }
    rng = np.random.default_rng(int(sampling_seed))
    summary = {
        "enabled": True,
        "max_cells_per_condition": int(max_cells_per_condition),
        "sampling_seed": int(sampling_seed),
        "conditions": {},
    }
    for split in ("train", "val"):
        for condition in pert_data.set2conditions.get(split, []):
            entries = pert_data.dataset_processed[condition]
            before = len(entries)
            if before > max_cells_per_condition:
                indexes = np.sort(rng.choice(before, size=max_cells_per_condition, replace=False))
                pert_data.dataset_processed[condition] = [entries[int(index)] for index in indexes]
            summary["conditions"][str(condition)] = {
                "split": split,
                "before": int(before),
                "after": int(len(pert_data.dataset_processed[condition])),
            }
    for condition in pert_data.set2conditions.get("test", []):
        summary["conditions"][str(condition)] = {
            "split": "test",
            "before": int(len(pert_data.dataset_processed[condition])),
            "after": int(len(pert_data.dataset_processed[condition])),
        }
    return summary


def _write_prediction_records(
    test_res: dict,
    ctrl_mean: np.ndarray,
    gene_names: list[str],
    dataset: str,
    seed: int,
    split: str,
    out: Path,
    run_type: str = "formal",
) -> pd.DataFrame:
    pred_effects: dict[str, np.ndarray] = {}
    true_effects: dict[str, np.ndarray] = {}
    rows: list[dict] = []
    pert_cat = np.asarray(test_res["pert_cat"])
    gene_panel_id = f"gears::{dataset}::{split}::n_genes_{len(gene_names)}"
    gene_hash = _gene_order_hash(gene_names)
    for pert in sorted(np.unique(pert_cat)):
        idx = np.where(pert_cat == pert)[0]
        pred_raw = np.asarray(test_res["pred"][idx], dtype=np.float32).mean(axis=0)
        true_raw = np.asarray(test_res["truth"][idx], dtype=np.float32).mean(axis=0)
        pred_effect = pred_raw - ctrl_mean
        true_effect = true_raw - ctrl_mean
        rmse = float(np.sqrt(np.mean((pred_effect - true_effect) ** 2)))
        denom = float(np.linalg.norm(pred_effect) * np.linalg.norm(true_effect) + 1e-8)
        cosine_error = float(1.0 - np.dot(pred_effect, true_effect) / denom)
        gears_uncertainty = np.nan
        if "logvar" in test_res:
            logvar = np.asarray(test_res["logvar"][idx], dtype=np.float32).mean(axis=0)
            # GEARS.predict reports exp(-mean(logvar)) as a confidence-like
            # score; for long-form confidence evaluation we store both the
            # raw mean log variance and an aligned confidence proxy.
            gears_uncertainty = float(np.mean(logvar))
            gears_confidence = float(np.exp(-gears_uncertainty))
        else:
            gears_confidence = np.nan
        record_id = f"GEARS::{dataset}::seed{seed}::{split}::{pert}"
        pred_key = record_id + "::pred"
        true_key = record_id + "::true"
        pred_effects[pred_key] = pred_effect.astype(np.float32)
        true_effects[true_key] = true_effect.astype(np.float32)
        rows.append(
            {
                "record_id": record_id,
                "task_id": pert,
                "task_key": f"{dataset}::{pert}",
                "dataset_name": dataset,
                "dataset_group": _dataset_group(dataset),
                "fold_id": int(seed),
                "split": "test",
                "context": f"GEARS_{split}_heldout",
                "perturbation": str(pert),
                "predictor_name": "GEARS",
                "schema_version": "safeconf_prediction_record_v1",
                "run_type": run_type,
                "gene_panel_id": gene_panel_id,
                "gene_order_hash": gene_hash,
                "effect_definition": "mean_diff",
                "normalization_id": "gears_mean_expression_minus_ctrl_v1",
                "error_normalization": "raw_rmse",
                "predicted_effect_key": pred_key,
                "true_effect_key": true_key,
                "true_error_rmse": rmse,
                "true_error_cosine": cosine_error,
                "gears_uncertainty_logvar_mean": gears_uncertainty,
                "gears_uncertainty_confidence": gears_confidence,
                "n_cells": int(len(idx)),
            }
        )
    rec = pd.DataFrame(rows)
    rec.to_csv(out / "PREDICTION_RECORDS.csv", index=False)
    np.savez_compressed(out / "gears_predicted_effects.npz", **pred_effects)
    np.savez_compressed(out / "gears_true_effects.npz", **true_effects)
    return rec


def _prediction_only_forward(loader, model, uncertainty: bool, device: str) -> dict:
    """Run the test model without reading ``batch.y`` or truth-derived fields."""
    model.eval()
    model.to(device)
    perturbations: list[str] = []
    predictions: list[torch.Tensor] = []
    logvars: list[torch.Tensor] = []
    for batch in loader:
        perturbations.extend(map(str, batch.pert))
        # Moving the full PyG Batch would also copy ``y`` and ``de_idx`` to the
        # accelerator.  GEARS_Model.forward only consumes x and batch, so the
        # pretruth pass receives an explicit truth-free view.
        prediction_view = SimpleNamespace(
            x=batch.x.to(device),
            batch=batch.batch.to(device),
        )
        with torch.no_grad():
            if uncertainty:
                prediction, logvar = model(prediction_view)
                logvars.extend(logvar.detach().cpu())
            else:
                prediction = model(prediction_view)
        predictions.extend(prediction.detach().cpu())
    result = {
        "pert_cat": np.asarray(perturbations, dtype=object),
        "pred": torch.stack(predictions).numpy(),
        "_cell_task_order_sha256": _string_sequence_hash(perturbations),
    }
    if uncertainty:
        result["logvar"] = torch.stack(logvars).numpy()
    return result


def _write_pretruth_score_lock(
    prediction_res: dict,
    ctrl_mean: np.ndarray,
    gene_names: list[str],
    dataset: str,
    seed: int,
    split: str,
    tables: Path,
    arrays: Path,
) -> tuple[pd.DataFrame, dict]:
    """Persist deployment scores and prediction vectors before truth unlock."""
    pert_cat = np.asarray(prediction_res["pert_cat"], dtype=object)
    gene_hash = _gene_order_hash(gene_names)
    predicted_effects: dict[str, np.ndarray] = {}
    rows: list[dict] = []
    for perturbation in sorted(np.unique(pert_cat)):
        indexes = np.where(pert_cat == perturbation)[0]
        predicted_raw = np.asarray(prediction_res["pred"][indexes], dtype=np.float32).mean(axis=0)
        predicted_effect = predicted_raw - np.asarray(ctrl_mean, dtype=np.float32)
        logvar_mean = np.nan
        if "logvar" in prediction_res:
            logvar_mean = float(
                np.asarray(prediction_res["logvar"][indexes], dtype=np.float32).mean()
            )
        key = f"GEARS::{dataset}::seed{seed}::{split}::{perturbation}::pred"
        predicted_effects[key] = predicted_effect.astype(np.float32)
        rows.append(
            {
                "record_id": key.rsplit("::pred", 1)[0],
                "task_id": str(perturbation),
                "task_key": f"{dataset}::{perturbation}",
                "dataset_name": dataset,
                "fold_id": int(seed),
                "split": "test",
                "predictor_name": "GEARS-UQ",
                "gene_order_hash": gene_hash,
                "predicted_effect_key": key,
                "predicted_effect_rms_magnitude": float(
                    np.sqrt(np.mean(np.square(predicted_effect, dtype=np.float64)))
                ),
                "gears_uncertainty_logvar_mean": logvar_mean,
                "n_cells": int(len(indexes)),
                "truth_fields_accessed": False,
            }
        )
    frame = pd.DataFrame(rows).sort_values("task_id", kind="stable").reset_index(drop=True)
    score_csv = tables / "PRETRUTH_SCORE_LOCK.csv"
    prediction_npz = arrays / "gears_predicted_effects.npz"
    frame.to_csv(score_csv, index=False)
    np.savez_compressed(prediction_npz, **predicted_effects)
    lock = {
        "schema": "safeconf_gears_pretruth_score_lock_v1",
        "locked_at": datetime.now().astimezone().isoformat(timespec="microseconds"),
        "dataset": dataset,
        "seed": int(seed),
        "split": split,
        "n_tasks": int(len(frame)),
        "n_test_cells": int(len(pert_cat)),
        "prediction_cell_task_order_sha256": str(
            prediction_res["_cell_task_order_sha256"]
        ),
        "gene_order_hash": gene_hash,
        "truth_fields_accessed": False,
        "score_csv": str(score_csv),
        "score_csv_sha256": _sha256(score_csv),
        "prediction_npz": str(prediction_npz),
        "prediction_npz_sha256": _sha256(prediction_npz),
    }
    lock_path = tables / "PRETRUTH_SCORE_LOCK.json"
    lock_path.write_text(
        json.dumps(lock, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lock["lock_json_sha256"] = _sha256(lock_path)
    return frame, lock


def _truth_unlock(
    loader,
    prediction_res: dict,
    device: str,
) -> dict:
    """Read test truth only after the score lock has been durably written."""
    expected_perturbations = np.asarray(prediction_res["pert_cat"], dtype=object)
    observed_perturbations: list[str] = []
    truths: list[torch.Tensor] = []
    prediction_de: list[torch.Tensor] = []
    truth_de: list[torch.Tensor] = []
    cursor = 0
    prediction_tensor = torch.as_tensor(prediction_res["pred"])
    for batch in loader:
        batch.to(device)
        batch_size = len(batch.pert)
        observed_perturbations.extend(map(str, batch.pert))
        truth = batch.y.detach().cpu()
        truths.extend(truth)
        batch_predictions = prediction_tensor[cursor : cursor + batch_size]
        for row_index, de_index in enumerate(batch.de_idx):
            prediction_de.append(batch_predictions[row_index, de_index])
            truth_de.append(truth[row_index, de_index])
        cursor += batch_size
    observed = np.asarray(observed_perturbations, dtype=object)
    if not np.array_equal(expected_perturbations, observed):
        raise RuntimeError("test loader order changed between score lock and truth unlock")
    result = dict(prediction_res)
    result["_truth_cell_task_order_sha256"] = _string_sequence_hash(observed)
    result["truth"] = torch.stack(truths).numpy()
    result["pred_de"] = torch.stack(prediction_de).numpy()
    result["truth_de"] = torch.stack(truth_de).numpy()
    return result


def _write_locked_prediction_records(
    score_lock: pd.DataFrame,
    test_res: dict,
    ctrl_mean: np.ndarray,
    gene_names: list[str],
    dataset: str,
    seed: int,
    split: str,
    tables: Path,
    arrays: Path,
    run_type: str,
) -> pd.DataFrame:
    """Combine immutable pretruth scores with truth read after the lock."""
    pert_cat = np.asarray(test_res["pert_cat"], dtype=object)
    score_by_task = score_lock.set_index("task_id", verify_integrity=True)
    with np.load(arrays / "gears_predicted_effects.npz") as prediction_archive:
        prediction_arrays = {
            key: np.asarray(prediction_archive[key], dtype=np.float32)
            for key in prediction_archive.files
        }
    true_effects: dict[str, np.ndarray] = {}
    rows: list[dict] = []
    for perturbation in sorted(np.unique(pert_cat)):
        indexes = np.where(pert_cat == perturbation)[0]
        truth_raw = np.asarray(test_res["truth"][indexes], dtype=np.float32).mean(axis=0)
        true_effect = truth_raw - np.asarray(ctrl_mean, dtype=np.float32)
        locked = score_by_task.loc[str(perturbation)]
        prediction_key = str(locked["predicted_effect_key"])
        predicted_effect = prediction_arrays[prediction_key]
        true_key = prediction_key.rsplit("::pred", 1)[0] + "::true"
        true_effects[true_key] = true_effect.astype(np.float32)
        rmse = float(
            np.sqrt(np.mean(np.square(predicted_effect - true_effect, dtype=np.float64)))
        )
        denominator = float(np.linalg.norm(predicted_effect) * np.linalg.norm(true_effect))
        cosine_error = (
            float(1.0 - np.dot(predicted_effect, true_effect) / denominator)
            if denominator > 1e-12
            else np.nan
        )
        logvar_mean = float(locked["gears_uncertainty_logvar_mean"])
        rows.append(
            {
                "record_id": str(locked["record_id"]),
                "task_id": str(perturbation),
                "task_key": str(locked["task_key"]),
                "dataset_name": dataset,
                "dataset_group": _dataset_group(dataset),
                "fold_id": int(seed),
                "split": "test",
                "context": f"GEARS_{split}_heldout",
                "perturbation": str(perturbation),
                "predictor_name": "GEARS",
                "schema_version": "safeconf_prediction_record_v1",
                "run_type": run_type,
                "gene_panel_id": f"gears::{dataset}::{split}::n_genes_{len(gene_names)}",
                "gene_order_hash": _gene_order_hash(gene_names),
                "effect_definition": "mean_diff",
                "normalization_id": "gears_mean_expression_minus_ctrl_v1",
                "error_normalization": "raw_rmse",
                "predicted_effect_key": prediction_key,
                "true_effect_key": true_key,
                "true_error_rmse": rmse,
                "true_error_cosine": cosine_error,
                "gears_uncertainty_logvar_mean": logvar_mean,
                "gears_uncertainty_confidence": float(np.exp(-logvar_mean)),
                "predicted_effect_rms_magnitude": float(
                    locked["predicted_effect_rms_magnitude"]
                ),
                "n_cells": int(locked["n_cells"]),
                "pretruth_score_lock_sha256": str(
                    json.loads(
                        (tables / "PRETRUTH_SCORE_LOCK.json").read_text(encoding="utf-8")
                    )["score_csv_sha256"]
                ),
                "pretruth_prediction_npz_sha256": _sha256(
                    arrays / "gears_predicted_effects.npz"
                ),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(tables / "PREDICTION_RECORDS.csv", index=False)
    np.savez_compressed(arrays / "gears_true_effects.npz", **true_effects)
    return frame


def run_one(args, out_dir: Path) -> dict:
    import gears
    from gears.inference import compute_metrics, evaluate

    dataset = args.dataset
    _set_all_random_seeds(args.seed)
    row = {
        "dataset": dataset,
        "split": args.split,
        "seed": args.seed,
        "epochs": args.epochs,
        "run_type": args.run_type,
        "require_cuda": bool(args.require_cuda),
        "strict_score_lock_before_truth": bool(args.strict_score_lock_before_truth),
        "rng_control": {
            "python": int(args.seed),
            "numpy": int(args.seed),
            "torch_cpu": int(args.seed),
            "torch_cuda_all": int(args.seed),
        },
        "status": "started",
    }
    try:
        run_out = out_dir / dataset / f"seed_{args.seed}"
        tables = run_out / "tables"
        arrays = run_out / "arrays"
        logs = run_out / "logs"
        for path in [tables, arrays, logs]:
            path.mkdir(parents=True, exist_ok=True)

        data_root = Path(args.data_path)
        pert_data = gears.PertData(str(data_root))
        processed_dir = data_root / f"{dataset}_local_atlas"
        processed_file = processed_dir / "perturb_processed.h5ad"
        if args.reuse_processed_local and processed_file.exists():
            pert_data.load(data_path=str(processed_dir))
        elif args.use_local_atlas and dataset in LOCAL_ATLAS_FILES:
            adata = load_local_gears_adata(dataset, max_genes=args.max_genes)
            pert_data.new_data_process(f"{dataset}_local_atlas", adata=adata)
        else:
            pert_data.load(data_name=dataset)
        _set_all_random_seeds(args.seed)
        pert_data.prepare_split(
            split=args.split,
            seed=args.seed,
            train_gene_set_size=args.train_gene_set_size,
        )
        if args.test_perturbations_file is not None:
            available_conditions = set(pert_data.adata.obs["condition"].astype(str).unique())
            requested_test_conditions = _read_requested_test_conditions(
                args.test_perturbations_file,
                available_conditions,
            )
            fixed_split_info = _override_fixed_test_conditions(
                pert_data,
                requested_test_conditions,
                deterministic_val=args.fixed_test_deterministic_val,
            )
            row.update(
                {
                    "fixed_test_perturbations_file": str(args.test_perturbations_file),
                    **fixed_split_info,
                }
            )
        if args.split != "no_test" and "val" not in getattr(pert_data, "set2conditions", {}):
            train_conditions = list(pert_data.set2conditions.get("train", []))
            val_pool = [c for c in train_conditions if c != "ctrl"] or train_conditions
            n_val = max(1, min(len(val_pool) // 5 or 1, len(val_pool)))
            pert_data.set2conditions["val"] = val_pool[:n_val]
            pert_data.set2conditions["train"] = [c for c in train_conditions if c not in pert_data.set2conditions["val"]] or train_conditions

        sampling_info = _subsample_train_val_graphs(
            pert_data,
            max_cells_per_condition=args.max_cells_per_condition,
            sampling_seed=args.condition_sampling_seed,
        )

        _set_all_random_seeds(args.seed)
        pert_data.get_dataloader(batch_size=args.batch_size, test_batch_size=args.test_batch_size)
        if "test_loader" in getattr(pert_data, "dataloader", {}):
            # GEARS 0.0.2 accepts ``test_batch_size`` but internally builds the
            # test loader with ``batch_size``.  Rebuild it explicitly so the
            # requested and recorded test batching contract is real.
            try:
                from torch_geometric.loader import DataLoader as GeometricDataLoader
            except ImportError:
                from torch_geometric.data import DataLoader as GeometricDataLoader

            original_test_loader = pert_data.dataloader["test_loader"]
            pert_data.dataloader["test_loader"] = GeometricDataLoader(
                original_test_loader.dataset,
                batch_size=args.test_batch_size,
                shuffle=False,
            )
            actual_test_batch_size = int(
                pert_data.dataloader["test_loader"].batch_size
            )
            if actual_test_batch_size != int(args.test_batch_size):
                raise RuntimeError(
                    "GEARS test batch-size contract failed: "
                    f"requested={args.test_batch_size}, actual={actual_test_batch_size}"
                )
            row["requested_test_batch_size"] = int(args.test_batch_size)
            row["actual_test_batch_size"] = actual_test_batch_size
        device = args.device
        if device.startswith("cuda") and args.require_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was required but torch.cuda.is_available() is false")
            requested_index = int(device.split(":", 1)[1]) if ":" in device else 0
            if requested_index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"CUDA device {device} was required but only "
                    f"{torch.cuda.device_count()} device(s) are visible"
                )
        elif device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        model = gears.GEARS(pert_data, device=device, weight_bias_track=False)
        _set_all_random_seeds(args.seed)
        model.model_initialize(
            hidden_size=args.hidden_size,
            num_go_gnn_layers=1,
            num_gene_gnn_layers=1,
            decoder_hidden_size=args.decoder_hidden_size,
            num_similar_genes_go_graph=args.num_similar_genes,
            num_similar_genes_co_express_graph=args.num_similar_genes,
            coexpress_threshold=args.coexpress_threshold,
            uncertainty=args.uncertainty,
            direction_lambda=args.direction_lambda,
        )
        initial_model_hash = _state_dict_sha256(model.model)
        if args.strict_score_lock_before_truth:
            test_loader = model.dataloader.pop("test_loader")
            try:
                model.train(epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay)
            finally:
                model.dataloader["test_loader"] = test_loader
            trained_model_hash = _state_dict_sha256(model.best_model)
        else:
            model.train(epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay)
            trained_model_hash = _state_dict_sha256(model.best_model)
        model_dir = run_out / "model"
        model.save_model(str(model_dir))
        saved_model_state_hash = _state_mapping_sha256(
            torch.load(model_dir / "model.pt", map_location="cpu")
        )
        if saved_model_state_hash != trained_model_hash:
            raise RuntimeError("saved GEARS checkpoint differs from trained best_model")

        ctrl_mean = _dense_mean(pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"].X)
        gene_names = list(pert_data.adata.var_names.astype(str))

        if args.strict_score_lock_before_truth:
            prediction_res = _prediction_only_forward(
                test_loader,
                model.best_model,
                args.uncertainty,
                device,
            )
            score_frame, score_lock = _write_pretruth_score_lock(
                prediction_res=prediction_res,
                ctrl_mean=ctrl_mean,
                gene_names=gene_names,
                dataset=dataset,
                seed=args.seed,
                split=args.split,
                tables=tables,
                arrays=arrays,
            )
            truth_unlock_started_at = datetime.now().astimezone().isoformat(
                timespec="microseconds"
            )
            test_res = _truth_unlock(test_loader, prediction_res, device)
            metrics, pert_metrics = compute_metrics(test_res)
            rec = _write_locked_prediction_records(
                score_lock=score_frame,
                test_res=test_res,
                ctrl_mean=ctrl_mean,
                gene_names=gene_names,
                dataset=dataset,
                seed=args.seed,
                split=args.split,
                tables=tables,
                arrays=arrays,
                run_type=args.run_type,
            )
            truth_unlock = {
                "schema": "safeconf_gears_truth_unlock_v1",
                "truth_unlock_started_at": truth_unlock_started_at,
                "pretruth_score_csv_sha256": score_lock["score_csv_sha256"],
                "pretruth_prediction_npz_sha256": score_lock[
                    "prediction_npz_sha256"
                ],
                "pretruth_lock_json_sha256": score_lock["lock_json_sha256"],
                "prediction_cell_task_order_sha256": score_lock[
                    "prediction_cell_task_order_sha256"
                ],
                "truth_cell_task_order_sha256": test_res[
                    "_truth_cell_task_order_sha256"
                ],
                "records_csv_sha256": _sha256(tables / "PREDICTION_RECORDS.csv"),
                "truth_npz_sha256": _sha256(arrays / "gears_true_effects.npz"),
                "n_tasks": int(len(rec)),
            }
            (tables / "TRUTH_UNLOCK.json").write_text(
                json.dumps(truth_unlock, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        else:
            test_res = evaluate(
                pert_data.dataloader["test_loader"],
                model.best_model,
                args.uncertainty,
                device,
            )
            metrics, pert_metrics = compute_metrics(test_res)
            rec = _write_prediction_records(
                test_res=test_res,
                ctrl_mean=ctrl_mean,
                gene_names=gene_names,
                dataset=dataset,
                seed=args.seed,
                split=args.split,
                out=tables,
                run_type=args.run_type,
            )
            # Legacy layout: write beside tables, then move under arrays.
            (tables / "gears_predicted_effects.npz").replace(
                arrays / "gears_predicted_effects.npz"
            )
            (tables / "gears_true_effects.npz").replace(
                arrays / "gears_true_effects.npz"
            )
        model_artifacts = {
            str(path.relative_to(run_out)): _sha256(path)
            for path in sorted(model_dir.glob("*"))
            if path.is_file()
        }
        cache_candidates = sorted(
            set(processed_dir.glob("*_co_expression_network.csv"))
            | set((processed_dir / "splits").glob("*.pkl"))
        )
        cache_artifacts = {
            str(path.relative_to(data_root)): _sha256(path)
            for path in cache_candidates
            if path.is_file()
        }
        actual_condition_sets = {
            split_name: list(map(str, pert_data.set2conditions.get(split_name, [])))
            for split_name in ("train", "val", "test")
        }
        critical_outputs = [
            tables / "PREDICTION_RECORDS.csv",
            arrays / "gears_predicted_effects.npz",
            arrays / "gears_true_effects.npz",
        ]
        if args.strict_score_lock_before_truth:
            critical_outputs.extend(
                [
                    tables / "PRETRUTH_SCORE_LOCK.csv",
                    tables / "PRETRUTH_SCORE_LOCK.json",
                    tables / "TRUTH_UNLOCK.json",
                ]
            )
        output_artifacts = {
            str(path.relative_to(run_out)): _sha256(path)
            for path in critical_outputs
        }
        pd.DataFrame([{"perturbation": k, **v} for k, v in pert_metrics.items()]).to_csv(
            tables / f"GEARS_{dataset}_{args.split}_PERT_METRICS.csv", index=False
        )
        row.update({f"test_{k}": float(v) for k, v in metrics.items()})
        row.update(
            {
                "status": "ok",
                "n_prediction_records": int(len(rec)),
                "records_csv": str(tables / "PREDICTION_RECORDS.csv"),
                "predicted_npz": str(arrays / "gears_predicted_effects.npz"),
                "true_npz": str(arrays / "gears_true_effects.npz"),
                "condition_graph_sampling": sampling_info,
                "actual_condition_sets": actual_condition_sets,
                "actual_condition_sets_sha256": _string_sequence_hash(
                    [
                        f"{split_name}:{condition}"
                        for split_name in ("train", "val", "test")
                        for condition in actual_condition_sets[split_name]
                    ]
                ),
                "data_path": str(data_root.resolve()),
                "actual_device": device,
                "initial_model_state_sha256": initial_model_hash,
                "trained_model_state_sha256": trained_model_hash,
                "saved_model_state_sha256": saved_model_state_hash,
                "gene_order_hash": _gene_order_hash(gene_names),
                "n_genes": int(len(gene_names)),
                "test_manifest_sha256": (
                    _sha256(args.test_perturbations_file)
                    if args.test_perturbations_file is not None
                    else None
                ),
                "runner_sha256": _sha256(Path(__file__).resolve()),
                "training_contract": {
                    "epochs": int(args.epochs),
                    "hidden_size": int(args.hidden_size),
                    "decoder_hidden_size": int(args.decoder_hidden_size),
                    "num_similar_genes": int(args.num_similar_genes),
                    "batch_size": int(args.batch_size),
                    "requested_test_batch_size": int(args.test_batch_size),
                    "max_cells_per_condition": int(args.max_cells_per_condition),
                    "condition_sampling_seed": int(args.condition_sampling_seed),
                    "lr": float(args.lr),
                    "weight_decay": float(args.weight_decay),
                    "coexpress_threshold": float(args.coexpress_threshold),
                    "direction_lambda": float(args.direction_lambda),
                    "uncertainty": bool(args.uncertainty),
                    "fixed_test_deterministic_val": bool(
                        args.fixed_test_deterministic_val
                    ),
                    "train_gene_set_size": float(args.train_gene_set_size),
                    "max_genes": int(args.max_genes),
                },
                "model_artifact_sha256": model_artifacts,
                "cache_artifact_sha256": cache_artifacts,
                "critical_output_sha256": output_artifacts,
            }
        )
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = repr(exc)
        row["traceback"] = traceback.format_exc(limit=8)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate GEARS and export per-prediction records.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/gears_prediction_records"))
    parser.add_argument("--data-path", default="/home/yyf/data/gears_formal_baselines_v2")
    parser.add_argument("--dataset", default="norman", choices=sorted(LOCAL_ATLAS_FILES))
    parser.add_argument("--split", default="single")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run-type", choices=["smoke", "formal"], default="formal")
    parser.add_argument(
        "--test-perturbations-file",
        type=Path,
        default=None,
        help="Optional fixed test perturbation list. Plain gene names are mapped to GENE+ctrl when available.",
    )
    parser.add_argument(
        "--fixed-test-deterministic-val",
        action="store_true",
        help="When using --test-perturbations-file, choose validation conditions deterministically from the remaining train pool.",
    )
    parser.add_argument("--train-gene-set-size", type=float, default=0.75)
    parser.add_argument("--use-local-atlas", action="store_true", default=True)
    parser.add_argument("--reuse-processed-local", action="store_true", default=True)
    parser.add_argument("--max-genes", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-batch-size", type=int, default=64)
    parser.add_argument(
        "--max-cells-per-condition",
        type=int,
        default=0,
        help="Deterministic maximum graph count per train/val condition; 0 retains all graphs. Test graphs are never sampled.",
    )
    parser.add_argument(
        "--condition-sampling-seed",
        type=int,
        default=20260766,
        help="Seed for train/val condition-stratified graph sampling.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--decoder-hidden-size", type=int, default=16)
    parser.add_argument("--num-similar-genes", type=int, default=10)
    parser.add_argument("--coexpress-threshold", type=float, default=0.4)
    parser.add_argument("--direction-lambda", type=float, default=0.1)
    parser.add_argument("--uncertainty", action="store_true")
    parser.add_argument(
        "--strict-score-lock-before-truth",
        action="store_true",
        help=(
            "Train without a test loader, write prediction/logvar/magnitude and "
            "their hashes, then read batch.y in a separate truth-unlock pass."
        ),
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail closed instead of silently falling back to CPU.",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    row = run_one(args, args.out_dir)
    status_path = args.out_dir / "GEARS_PREDICTION_RECORD_STATUS.csv"
    existing = pd.read_csv(status_path) if status_path.exists() else pd.DataFrame()
    pd.concat([existing, pd.DataFrame([row])], ignore_index=True).to_csv(status_path, index=False)
    (args.out_dir / "GEARS_PREDICTION_RECORD_STATUS.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(row, ensure_ascii=False, indent=2))
    if row.get("status") != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
