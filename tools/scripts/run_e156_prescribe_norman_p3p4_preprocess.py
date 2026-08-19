#!/usr/bin/env python3
"""E156: development-only PRESCRIBE preprocessing for frozen Norman P3/P4.

The fitting path is restricted to the shared E155 training conditions.  The
raw container and its observation metadata are inspected, but P3/P4 test-X rows
are never indexed, materialized, or transformed.  Runtime assets contain train
and validation cells only.  Test X remains unavailable until E157 has hashed
both checkpoints and truth-free task scores.  No model is built, trained,
scored, or evaluated in this program.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.utils.extmath import safe_sparse_dot


ROOT = Path(__file__).resolve().parents[2]
RAW = Path(
    "/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/"
    "NormanWeissman2019_filtered.h5ad"
)
PRESCRIBE = Path("/home/yyf/archive/external/PRESCRIBE")
E155 = ROOT / "docs/实验结果/E155_prescribe_norman_p3p4_contract_20260714"
OUT = ROOT / "docs/实验结果/E156_prescribe_norman_p3p4_preprocess_20260714"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"
DATA_ROOT = Path("/home/yyf/data/safeconf_e156_prescribe")
SHARED = DATA_ROOT / "shared_train_fit"

SEED = 3407
TARGET_SUM = 10_000.0
MIN_TRAIN_DETECTED_CELLS = 50
N_HVG = 2_000
N_PCA = 10
PANELS = ("Norman_P3", "Norman_P4")
PANEL_SLUG = {"Norman_P3": "norman_p3", "Norman_P4": "norman_p4"}
EXPECTED_E155_COMMIT = "53222bc"
EXPECTED_RAW_SHA256 = "efde6f5301fe256725dce1d980f37bd96a13481a9a16135515897368e631affc"
EXPECTED_UPSTREAM_COMMIT = "6f7264a205aaff654a9594863c5c10b656f88ebe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-cell-graphs",
        action="store_true",
        help="Stop after H5AD/priors; default also materializes official PertData graphs.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def runner_git_provenance() -> dict[str, object]:
    runner = Path(__file__).resolve()
    relative = runner.relative_to(ROOT)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    committed = subprocess.check_output(
        ["git", "show", f"HEAD:{relative.as_posix()}"], cwd=ROOT
    )
    committed_sha256 = hashlib.sha256(committed).hexdigest()
    working_sha256 = sha256_file(runner)
    if committed_sha256 != working_sha256:
        raise RuntimeError(
            "E156 runner differs from the committed HEAD blob; commit the runner before formal execution"
        )
    return {
        "runner_path": str(runner),
        "runner_sha256": working_sha256,
        "runner_git_head": head,
        "runner_matches_git_head_blob": True,
    }


def normalize_condition(value: str) -> str:
    parts = str(value).replace("control", "ctrl").split("_")
    if len(parts) == 1 and parts[0] == "ctrl":
        return "ctrl"
    if len(parts) == 1:
        parts.append("ctrl")
    return "+".join(sorted(parts))


def perturbation_genes(condition: str) -> tuple[str, ...]:
    return tuple(gene for gene in str(condition).split("+") if gene != "ctrl")


def write_status(**updates: object) -> dict[str, object]:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "RUN_STATUS.json"
    current: dict[str, object] = {}
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
    current.update(updates)
    path.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return current


def validate_e155() -> tuple[dict[str, object], dict[str, pd.DataFrame], dict[str, dict[str, list[str]]]]:
    status_path = E155 / "RUN_STATUS.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("phase") != "contract_frozen_predictions_and_errors_unseen":
        raise RuntimeError("E155 is not in the frozen/unseen phase")
    if status.get("model_and_preprocess_seed") != SEED:
        raise RuntimeError("E155 seed changed")
    if not str(status.get("primary_preprocessing", "")).startswith("train-only"):
        raise RuntimeError("E155 does not require train-only preprocessing")
    for relative, expected in status["artifact_sha256"].items():
        path = E155 / relative
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"E155 hash mismatch: {relative}: {observed}")

    commit = subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", str(E155.relative_to(ROOT))],
        cwd=ROOT,
        text=True,
    ).strip()
    if not commit.startswith(EXPECTED_E155_COMMIT):
        raise RuntimeError(f"Unexpected E155 commit: {commit}")

    source_hashes = pd.read_csv(E155 / "manifests/E155_SOURCE_HASHES.csv")
    source_lookup = source_hashes.set_index("source_role")
    raw_expected = source_lookup.loc[
        "Norman_raw_metadata_and_future_expression_source", "sha256"
    ]
    if raw_expected != EXPECTED_RAW_SHA256 or sha256_file(RAW) != raw_expected:
        raise RuntimeError("Raw Norman source hash changed")
    upstream_commit = subprocess.check_output(
        ["git", "-C", str(PRESCRIBE), "rev-parse", "HEAD"], text=True
    ).strip()
    if upstream_commit != EXPECTED_UPSTREAM_COMMIT:
        raise RuntimeError(f"PRESCRIBE upstream commit changed: {upstream_commit}")
    for role in ["PRESCRIBE_Step1_preprocess", "PRESCRIBE_data_loader_worktree"]:
        row = source_lookup.loc[role]
        path = Path(row["path"])
        if sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"Frozen PRESCRIBE source changed: {role}")

    split_frames: dict[str, pd.DataFrame] = {}
    split_dicts: dict[str, dict[str, list[str]]] = {}
    for panel in PANELS:
        csv_path = E155 / "manifests" / f"{panel}_SPLIT.csv"
        pkl_path = E155 / "manifests" / f"{panel}_set2conditions.pkl"
        frame = pd.read_csv(csv_path)
        with pkl_path.open("rb") as handle:
            frozen = pickle.load(handle)
        csv_dict = {
            role: frame.loc[frame["split"].eq(role), "condition"].astype(str).tolist()
            for role in ["train", "val", "test"]
        }
        if {key: sorted(value) for key, value in csv_dict.items()} != {
            key: sorted(value) for key, value in frozen.items()
        }:
            raise RuntimeError(f"{panel}: CSV/pickle split mismatch")
        split_frames[panel] = frame
        split_dicts[panel] = frozen

    if set(split_dicts["Norman_P3"]["train"]) != set(split_dicts["Norman_P4"]["train"]):
        raise RuntimeError("P3/P4 training conditions are no longer shared")
    if set(split_dicts["Norman_P3"]["val"]) != set(split_dicts["Norman_P4"]["val"]):
        raise RuntimeError("P3/P4 validation conditions are no longer shared")
    return status, split_frames, split_dicts


def fixed_log_normalize(counts: sp.spmatrix) -> sp.csr_matrix:
    matrix = counts.tocsr().astype(np.float32, copy=True)
    totals = np.asarray(matrix.sum(axis=1)).reshape(-1)
    if np.any(~np.isfinite(totals)) or np.any(totals <= 0):
        raise RuntimeError("Zero/non-finite library-size cell in development input")
    scale = np.asarray(TARGET_SUM / totals, dtype=np.float32)
    matrix = sp.diags(scale, format="csr") @ matrix
    np.log1p(matrix.data, out=matrix.data)
    matrix.eliminate_zeros()
    return matrix.astype(np.float32)


def pca_transform(matrix: sp.spmatrix, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    transformed = safe_sparse_dot(matrix, components.T, dense_output=True)
    transformed -= np.asarray(mean @ components.T, dtype=transformed.dtype)
    return np.asarray(transformed, dtype=np.float32)


def load_raw_train(train_conditions: set[str]) -> tuple[ad.AnnData, pd.Series]:
    raw = sc.read_h5ad(RAW, backed="r")
    try:
        normalized = raw.obs["perturbation"].astype(str).map(normalize_condition)
        mask = normalized.isin(train_conditions).to_numpy()
        train = raw[mask].to_memory()
        train.obs["condition"] = pd.Categorical(normalized.loc[mask].to_numpy())
        train.obs["e156_split"] = pd.Categorical(["train"] * train.n_obs)
        return train, normalized
    finally:
        raw.file.close()


def build_shared_fit(
    split_dicts: dict[str, dict[str, list[str]]]
) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    train_conditions = set(split_dicts["Norman_P3"]["train"])
    all_frozen_conditions = set()
    for panel in PANELS:
        for values in split_dicts[panel].values():
            all_frozen_conditions.update(values)
    forced_genes = {
        gene for condition in all_frozen_conditions for gene in perturbation_genes(condition)
    }

    train, _ = load_raw_train(train_conditions)
    try:
        if not sp.issparse(train.X):
            train.X = sp.csr_matrix(train.X)
        detected = np.asarray((train.X > 0).sum(axis=0)).reshape(-1).astype(int)
        train_filter = detected >= MIN_TRAIN_DETECTED_CELLS
        if train_filter.sum() < N_HVG:
            raise RuntimeError("Fewer than 2,000 train-qualified genes")

        hvg_data = train[:, train_filter].copy()
        hvg_data.layers["counts"] = hvg_data.X.copy()
        sc.pp.highly_variable_genes(
            hvg_data,
            n_top_genes=N_HVG,
            flavor="seurat_v3",
            layer="counts",
            subset=False,
        )
        hvg_set = set(hvg_data.var_names[hvg_data.var["highly_variable"]])
        missing_forced = sorted(forced_genes - set(train.var_names.astype(str)))
        if missing_forced:
            raise RuntimeError(f"Frozen perturbation genes absent from raw var: {missing_forced}")
        model_set = hvg_set | forced_genes
        model_genes = [gene for gene in train.var_names.astype(str) if gene in model_set]

        hvg_meta = hvg_data.var.reindex(train.var_names)
        full_axis = pd.DataFrame(
            {
                "raw_gene_index": np.arange(train.n_vars, dtype=int),
                "gene": train.var_names.astype(str),
                "train_detected_cells": detected,
                "passes_train_min50_detection": train_filter,
                "train_seurat_v3_hvg": train.var_names.astype(str).isin(hvg_set),
                "forced_by_frozen_perturbation_identity": train.var_names.astype(str).isin(forced_genes),
                "included_in_model_axis": train.var_names.astype(str).isin(model_set),
            }
        )
        for column in ["highly_variable_rank", "means", "variances", "variances_norm"]:
            full_axis[f"train_hvg_{column}"] = hvg_meta[column].to_numpy() if column in hvg_meta else np.nan
        index_lookup = {gene: idx for idx, gene in enumerate(model_genes)}
        full_axis["model_gene_index"] = full_axis["gene"].map(index_lookup).astype("Int64")

        train_model = train[:, model_genes].copy()
        train_counts = train_model.X.tocsr().astype(np.float32)
        train_norm = fixed_log_normalize(train_counts)
        dense = train_norm.toarray().astype(np.float32, copy=False)
        pca = PCA(n_components=N_PCA, svd_solver="randomized", random_state=SEED)
        train_pca = pca.fit_transform(dense).astype(np.float32)
        del dense

        condition = train.obs["condition"].astype(str)
        counts_by_condition = condition.value_counts().sort_index()
        n_balance = int(counts_by_condition.min())
        selected_positions: list[int] = []
        balance_rows: list[dict[str, object]] = []
        for label in sorted(train_conditions):
            positions = np.flatnonzero(condition.to_numpy() == label)
            local_seed = int(sha256_text(f"E156|{SEED}|{label}")[:16], 16)
            rng = np.random.default_rng(local_seed)
            chosen = np.sort(rng.choice(positions, size=n_balance, replace=False))
            selected_positions.extend(chosen.tolist())
            for position in chosen:
                balance_rows.append(
                    {
                        "condition": label,
                        "raw_cell_id": str(train.obs_names[position]),
                        "train_cells_before_balance": int(len(positions)),
                        "balanced_cells": n_balance,
                        "selection_seed_sha256_prefix": f"{local_seed:016x}",
                    }
                )
        balance = pd.DataFrame(balance_rows)
        balanced_positions = np.asarray(selected_positions, dtype=int)
        balanced_condition = condition.iloc[balanced_positions].to_numpy()
        balanced_pca = train_pca[balanced_positions]
        ctrl = balanced_pca[balanced_condition == "ctrl"]
        sigma_ctrl = float(
            pairwise_distances(ctrl, ctrl, metric="sqeuclidean").sum()
            / (len(ctrl) * (len(ctrl) - 1))
        )
        e_rows: list[dict[str, object]] = []
        for label in sorted(train_conditions):
            values = balanced_pca[balanced_condition == label]
            within = pairwise_distances(values, values, metric="sqeuclidean")
            sigma = float(within.sum() / (len(values) * (len(values) - 1)))
            if label == "ctrl":
                delta = sigma_ctrl
                edistance = 0.0
            else:
                cross = pairwise_distances(ctrl, values, metric="sqeuclidean")
                delta = float(cross.mean())
                edistance = float(2.0 * delta - sigma - sigma_ctrl)
            e_rows.append(
                {
                    "condition": label,
                    "n_train_cells_before_balance": int(counts_by_condition[label]),
                    "n_balanced": n_balance,
                    "delta_to_control": delta,
                    "sigma_within": sigma,
                    "sigma_control": sigma_ctrl,
                    "edistance_training_rank_label": edistance,
                    "fit_scope": "train_only",
                    "permutation_pvalue_computed": False,
                }
            )
        edistance = pd.DataFrame(e_rows)
        numeric = edistance[
            ["delta_to_control", "sigma_within", "sigma_control", "edistance_training_rank_label"]
        ].to_numpy(float)
        if not np.isfinite(numeric).all():
            raise RuntimeError("Non-finite train E-distance label")

        control_pca = train_pca[condition.to_numpy() == "ctrl"]
        shared = {
            "model_genes": np.asarray(model_genes, dtype=str),
            "pca_mean": np.asarray(pca.mean_, dtype=np.float32),
            "pca_components": np.asarray(pca.components_, dtype=np.float32),
            "pca_explained_variance": np.asarray(pca.explained_variance_, dtype=np.float32),
            "pca_explained_variance_ratio": np.asarray(pca.explained_variance_ratio_, dtype=np.float32),
            "control_pca_mean": np.asarray(control_pca.mean(axis=0), dtype=np.float64),
            "control_pca_cov": np.asarray(np.cov(control_pca, rowvar=False), dtype=np.float64),
            "train_cell_ids": np.asarray(train.obs_names.astype(str)),
            "train_pca": train_pca,
            "balanced_positions": balanced_positions,
            "train_counts_model": train_counts,
            "train_norm_model": train_norm,
            "model_var": train_model.var.copy(),
            "train_obs": train.obs.copy(),
        }
        return full_axis, shared, edistance, balance
    finally:
        del train
        gc.collect()


def write_shared_assets(
    full_axis: pd.DataFrame,
    shared: dict[str, np.ndarray],
    edistance: pd.DataFrame,
    balance: pd.DataFrame,
) -> None:
    SHARED.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    full_axis.to_csv(TABLES / "E156_FULL_GENE_AXIS_AUDIT.csv", index=False)
    edistance.to_csv(TABLES / "E156_TRAIN_EDISTANCE.csv", index=False)
    balance.to_csv(TABLES / "E156_BALANCED_TRAIN_CELLS.csv", index=False)
    np.savez_compressed(
        SHARED / "TRAIN_ONLY_PCA_MODEL.npz",
        model_genes=shared["model_genes"],
        mean=shared["pca_mean"],
        components=shared["pca_components"],
        explained_variance=shared["pca_explained_variance"],
        explained_variance_ratio=shared["pca_explained_variance_ratio"],
    )
    balanced = ad.AnnData(
        X=shared["train_norm_model"][shared["balanced_positions"]].copy(),
        obs=shared["train_obs"].iloc[shared["balanced_positions"]].copy(),
        var=shared["model_var"].copy(),
    )
    balanced.obs["condition"] = pd.Categorical(balance["condition"].to_numpy())
    balanced.obsm["X_pca"] = shared["train_pca"][shared["balanced_positions"]].copy()
    balanced.uns["fit_scope"] = "train_only_balanced_input_for_Edistance"
    balanced.uns["normalization"] = "per-cell fixed target_sum=10000 then log1p"
    balanced.write_h5ad(SHARED / "train_only_perturb_e_distance.h5ad", compression="gzip")


def panel_adata(
    panel: str,
    split_frame: pd.DataFrame,
    split_dict: dict[str, list[str]],
    full_axis: pd.DataFrame,
    shared: dict[str, np.ndarray],
    edistance: pd.DataFrame,
) -> tuple[ad.AnnData, pd.DataFrame]:
    split_map = split_frame.set_index("condition")["split"].to_dict()
    selected = set(split_dict["train"]) | set(split_dict["val"])
    model_genes = shared["model_genes"].astype(str).tolist()
    raw = sc.read_h5ad(RAW, backed="r")
    try:
        normalized = raw.obs["perturbation"].astype(str).map(normalize_condition)
        mask = normalized.isin(selected).to_numpy()
        data = raw[mask, model_genes].to_memory()
        conditions = normalized.loc[mask].to_numpy()
    finally:
        raw.file.close()

    if not sp.issparse(data.X):
        data.X = sp.csr_matrix(data.X)
    counts = data.X.tocsr().astype(np.float32)
    normalized_x = fixed_log_normalize(counts)
    data.layers["counts"] = counts
    data.X = normalized_x
    data.obs["perturbation"] = pd.Categorical(conditions)
    data.obs["condition"] = pd.Categorical(conditions)
    data.obs["e156_split"] = pd.Categorical([split_map[value] for value in conditions])
    if "cell_type" not in data.obs:
        data.obs["cell_type"] = pd.Categorical(data.obs["cell_line"].astype(str))
    data.obs["condition_name"] = pd.Categorical(
        data.obs["cell_line"].astype(str) + "_" + data.obs["condition"].astype(str)
    )
    data.var["gene_name"] = data.var_names.astype(str)
    axis = full_axis.set_index("gene").loc[model_genes]
    for source, target in [
        ("train_detected_cells", "e156_train_detected_cells"),
        ("passes_train_min50_detection", "e156_train_min50"),
        ("train_seurat_v3_hvg", "highly_variable"),
        ("forced_by_frozen_perturbation_identity", "e156_forced_perturbation_gene"),
        ("train_hvg_highly_variable_rank", "highly_variable_rank"),
        ("train_hvg_means", "means"),
        ("train_hvg_variances", "variances"),
        ("train_hvg_variances_norm", "variances_norm"),
    ]:
        data.var[target] = axis[source].to_numpy()

    data.obsm["X_pca"] = pca_transform(data.X, shared["pca_mean"], shared["pca_components"])
    data.uns["pca_mean"] = shared["pca_mean"]
    data.uns["pca_components"] = shared["pca_components"]
    data.uns["processed"] = True
    data.uns["hvg"] = {
        "flavor": "seurat_v3",
        "n_top_genes": N_HVG,
        "fit_scope": "shared_train_only",
    }
    data.uns["log1p"] = {"base": None}

    e_lookup = edistance.set_index("condition")
    y_d: dict[str, float] = {}
    y_s: dict[str, float] = {}
    y_n: dict[str, float] = {}
    for condition in sorted(selected):
        if condition in set(split_dict["train"]):
            y_d[condition] = float(e_lookup.loc[condition, "delta_to_control"])
            y_s[condition] = float(e_lookup.loc[condition, "sigma_within"])
            y_n[condition] = float(e_lookup.loc[condition, "edistance_training_rank_label"])
        else:
            # Required keys for upstream graph creation.  The model's validation/test
            # paths do not read these labels; NaN is a fail-fast sentinel if that changes.
            y_d[condition] = float("nan")
            y_s[condition] = float("nan")
            y_n[condition] = float("nan")
    data.uns["y_d"] = y_d
    data.uns["y_s"] = y_s
    data.uns["y_n"] = y_n

    ranked = full_axis.loc[full_axis["included_in_model_axis"]].copy()
    ranked["rank_sort"] = ranked["train_hvg_highly_variable_rank"].fillna(np.inf)
    callback_order = ranked.sort_values(["rank_sort", "raw_gene_index"])["gene"].to_numpy(str)
    data.uns["rank_genes_groups_cov_all"] = {
        name: callback_order for name in data.obs["condition_name"].cat.categories
    }
    data.uns["e156_callback_gene_order_policy"] = (
        "fixed train-HVG order placeholder for upstream callback compatibility; "
        "not condition DE and forbidden for biological top-DE evaluation"
    )
    data.uns["e156_provenance"] = {
        "experiment": "E156_prescribe_norman_p3p4_preprocess",
        "panel": panel,
        "raw_sha256": EXPECTED_RAW_SHA256,
        "normalization": "per-cell fixed target_sum=10000 then log1p",
        "gene_selection_fit_scope": "shared_train_only",
        "pca_fit_scope": "shared_train_only",
        "edistance_label_scope": "shared_train_only; val NaN sentinels",
        "test_expression_role": (
            "raw container and obs metadata inspected; test-X rows never indexed, "
            "materialized, or transformed in E156"
        ),
        "endpoint_or_prediction_computed": False,
    }

    rows = []
    observed_counts = pd.Series(conditions).value_counts()
    for role in ["train", "val"]:
        for condition in sorted(split_dict[role]):
            rows.append(
                {
                    "panel": panel,
                    "split": role,
                    "condition": condition,
                    "n_cells_input_or_metadata": int(observed_counts[condition]),
                    "expression_matrix_read": True,
                    "n_perturbation_genes": len(perturbation_genes(condition)),
                    "training_edistance_label_finite": bool(np.isfinite(y_n[condition])),
                    "expression_used_for_gene_selection": role == "train",
                    "expression_used_for_pca_fit": role == "train",
                    "expression_used_for_edistance_fit": role == "train",
                    "expression_role_if_not_train": "fixed_transform_validation" if role != "train" else "fit",
                }
            )
    condition_audit = pd.read_csv(E155 / "manifests/E155_CONDITION_AUDIT.csv").set_index(
        "condition"
    )
    for condition in sorted(split_dict["test"]):
        rows.append(
            {
                "panel": panel,
                "split": "test",
                "condition": condition,
                "n_cells_input_or_metadata": int(
                    condition_audit.loc[condition, "n_cells_obs_metadata"]
                ),
                "expression_matrix_read": False,
                "n_perturbation_genes": len(perturbation_genes(condition)),
                "training_edistance_label_finite": False,
                "expression_used_for_gene_selection": False,
                "expression_used_for_pca_fit": False,
                "expression_used_for_edistance_fit": False,
                "expression_role_if_not_train": "obs_metadata_only_test_X_not_indexed_or_materialized",
            }
        )
    return data, pd.DataFrame(rows)


def write_panel_assets(
    panel: str,
    split_frame: pd.DataFrame,
    split_dict: dict[str, list[str]],
    full_axis: pd.DataFrame,
    shared: dict[str, np.ndarray],
    edistance: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    slug = PANEL_SLUG[panel]
    panel_dir = DATA_ROOT / slug
    if panel_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {panel_dir}")
    panel_dir.mkdir(parents=True)
    (panel_dir / "data_pyg").mkdir()

    data, split_audit = panel_adata(
        panel, split_frame, split_dict, full_axis, shared, edistance
    )
    development = data
    development_conditions = set(split_dict["train"]) | set(split_dict["val"])
    for key in ["y_d", "y_s", "y_n"]:
        development.uns[key] = {
            condition: value
            for condition, value in development.uns[key].items()
            if condition in development_conditions
        }
    development.uns["rank_genes_groups_cov_all"] = {
        key: value
        for key, value in development.uns["rank_genes_groups_cov_all"].items()
        if key in set(development.obs["condition_name"].astype(str))
    }
    h5ad = panel_dir / "perturb_processed.h5ad"
    development.write_h5ad(h5ad, compression="gzip")
    del development
    gc.collect()

    source_edistance = SHARED / "train_only_perturb_e_distance.h5ad"
    os.link(source_edistance, panel_dir / "perturb_e_distance.h5ad")
    with (panel_dir / f"set2conditions_{SEED}.pkl").open("wb") as handle:
        pickle.dump(split_dict, handle, protocol=4)
    frozen_genes: list[str] = []
    for role in ["train", "val", "test"]:
        for condition in split_dict[role]:
            for gene in perturbation_genes(condition):
                if gene not in frozen_genes:
                    frozen_genes.append(gene)
    with (panel_dir / f"frozen_pert_gene_set_{SEED}.pkl").open("wb") as handle:
        pickle.dump(frozen_genes, handle, protocol=4)
    np.save(panel_dir / "data_pyg/mean.npy", shared["control_pca_mean"])
    np.save(panel_dir / "data_pyg/cov.npy", shared["control_pca_cov"])
    edistance.to_csv(panel_dir / "train_only_edistance_labels.csv", index=False)

    expected_link = PRESCRIBE / "data" / slug
    if expected_link.exists() or expected_link.is_symlink():
        raise FileExistsError(f"Refusing to replace PRESCRIBE data link: {expected_link}")
    expected_link.symlink_to(panel_dir, target_is_directory=True)

    summary = {
        "panel": panel,
        "dataset_name": slug,
        "n_cells_from_development_plus_test_metadata": int(
            split_audit["n_cells_input_or_metadata"].sum()
        ),
        "n_development_cells": int(
            split_audit.loc[
                split_audit["split"].isin(["train", "val"]),
                "n_cells_input_or_metadata",
            ].sum()
        ),
        "n_test_cells_metadata_only": int(
            split_audit.loc[
                split_audit["split"].eq("test"), "n_cells_input_or_metadata"
            ].sum()
        ),
        "n_genes": int(len(shared["model_genes"])),
        "n_train_conditions": len(split_dict["train"]),
        "n_val_conditions": len(split_dict["val"]),
        "n_test_conditions": len(split_dict["test"]),
        "h5ad": str(h5ad),
        "test_X_rows_indexed_materialized_or_transformed": False,
        "prescribe_symlink": str(expected_link),
    }
    return split_audit, summary


def build_cell_graphs(panel: str, split_dict: dict[str, list[str]]) -> dict[str, object]:
    slug = PANEL_SLUG[panel]
    old_cwd = Path.cwd()
    old_path = list(sys.path)
    try:
        os.chdir(PRESCRIBE)
        sys.path.insert(0, str(PRESCRIBE))
        from src.data.pertdata import LoadData  # noqa: PLC0415

        pert_data = LoadData(data_name=slug, seed=SEED, backbone=None)
        observed = {
            role: sorted(pert_data.set2conditions[role]) for role in ["train", "val", "test"]
        }
        expected = {role: sorted(split_dict[role]) for role in ["train", "val", "test"]}
        if observed != expected:
            raise RuntimeError(f"{panel}: PertData split mismatch")
        graph_keys = set(pert_data.dataset_processed)
        expected_keys = set(split_dict["train"]) | set(split_dict["val"])
        if graph_keys != expected_keys:
            raise RuntimeError(
                f"{panel}: graph keys mismatch: missing={sorted(expected_keys-graph_keys)}, "
                f"extra={sorted(graph_keys-expected_keys)}"
            )
        graph_counts = {key: len(value) for key, value in pert_data.dataset_processed.items()}
        del pert_data
        gc.collect()
    finally:
        sys.path[:] = old_path
        os.chdir(old_cwd)
    graph_path = DATA_ROOT / slug / "data_pyg/cell_graphs.pkl"
    if not graph_path.exists():
        raise FileNotFoundError(graph_path)
    return {
        "panel": panel,
        "n_graph_conditions": len(graph_counts),
        "n_cell_graphs": int(sum(graph_counts.values())),
        "cell_graphs_path": str(graph_path),
        "cell_graphs_sha256": sha256_file(graph_path),
    }


def leakage_audit(split_dicts: dict[str, dict[str, list[str]]]) -> pd.DataFrame:
    rows = []
    for panel in PANELS:
        test_genes = {
            gene
            for condition in split_dicts[panel]["test"]
            for gene in perturbation_genes(condition)
        }
        for gene in sorted(test_genes):
            train_hits = [
                condition
                for condition in split_dicts[panel]["train"]
                if gene in perturbation_genes(condition)
            ]
            val_hits = [
                condition
                for condition in split_dicts[panel]["val"]
                if gene in perturbation_genes(condition)
            ]
            rows.append(
                {
                    "panel": panel,
                    "heldout_test_gene": gene,
                    "train_condition_hits": ";".join(train_hits),
                    "val_condition_hits": ";".join(val_hits),
                    "n_train_hits": len(train_hits),
                    "n_val_hits": len(val_hits),
                    "development_leakage": bool(train_hits or val_hits),
                }
            )
    frame = pd.DataFrame(rows)
    if frame["development_leakage"].any():
        raise RuntimeError("Held-out gene leakage detected")
    return frame


def provenance_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "component": "obs/task membership",
                "source": str(RAW),
                "operation": "normalize labels; exact E155 condition membership",
                "fit_scope": "none",
                "test_role": "obs metadata only; test-X rows not indexed or materialized",
            },
            {
                "component": "gene detection + HVG",
                "source": "raw X of E155 shared train conditions",
                "operation": "train detection>=50; seurat_v3 top2000; union frozen perturbation identities",
                "fit_scope": "train only",
                "test_role": "test-X rows not indexed or materialized",
            },
            {
                "component": "normalization",
                "source": "each cell count vector",
                "operation": "fixed per-cell target_sum=10000, log1p",
                "fit_scope": "no cross-cell fitted parameter",
                "test_role": "same fixed row-wise transform; no threshold/selection",
            },
            {
                "component": "PCA",
                "source": "normalized shared-train X on frozen model gene axis",
                "operation": "randomized PCA10 seed3407",
                "fit_scope": "train only",
                "test_role": "components/mean transform only",
            },
            {
                "component": "E-distance rank label",
                "source": "balanced shared-train PCA coordinates",
                "operation": "squared-Euclidean energy distance to train control",
                "fit_scope": "train only",
                "test_role": "test X absent from E156; validation uses NaN sentinel",
            },
            {
                "component": "rank_genes_groups_cov_all compatibility field",
                "source": "train-HVG order",
                "operation": "fixed placeholder required by upstream test callback",
                "fit_scope": "train only",
                "test_role": "not test DE; forbidden as biological top-DE endpoint",
            },
            {
                "component": "validation Y and PCA targets",
                "source": "raw validation X",
                "operation": "fixed normalization and frozen PCA transform",
                "fit_scope": "none",
                "test_role": "test-X rows are not indexed, materialized, or transformed in E156",
            },
        ]
    )


def native_code_audit() -> pd.DataFrame:
    findings = [
        (
            PRESCRIBE / "Step1_preprocess.py",
            "Upstream Step1 filters genes, fits HVG and prepares DE before splitting; "
            "normalize_per_cell(None) also derives a global target from all cells.",
            "Do not reuse for the E156 primary assets; fit all cross-cell parameters on train only.",
        ),
        (
            PRESCRIBE / "src/nn/loss.py",
            "ListMLE sorts scores by y_n, then applies -sum(log_softmax(sorted_scores)); "
            "that sum is permutation-invariant, so y_n ordering has no mathematical effect.",
            "Keep train-only y_n for native interface compatibility, but do not claim it drives ranking learning.",
        ),
        (
            PRESCRIBE / "src/model/lightening_module.py",
            "Native test_step reconstructs both pred and truth from PCA10; it does not return raw graph y as truth.",
            "The frozen primary Pearson endpoint must use PCA10-reconstructed truth for P1/P2 comparability; raw normalized truth is sensitivity only.",
        ),
        (
            PRESCRIBE / "src/nn/model.py",
            "Forward prediction consumes control-expression x plus perturbation identity; held-out target y/y_pca are loss/evaluation fields.",
            "Serialize validation y/y_pca only; E158 may create test truth after E157 checkpoint and task scores are locked.",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "source": str(path),
                "sha256": sha256_file(path),
                "audited_finding": finding,
                "E156_decision": decision,
            }
            for path, finding, decision in findings
        ]
    )


def artifact_hash_table(paths: list[Path]) -> pd.DataFrame:
    rows = []
    seen: set[tuple[int, int]] = set()
    for path in sorted(set(paths), key=str):
        stat = path.stat()
        inode_key = (stat.st_dev, stat.st_ino)
        digest = sha256_file(path)
        rows.append(
            {
                "path": str(path),
                "bytes": stat.st_size,
                "sha256": digest,
                "hardlink_duplicate_of_previous": inode_key in seen,
            }
        )
        seen.add(inode_key)
    return pd.DataFrame(rows)


def report_text(
    summaries: list[dict[str, object]],
    graph_summaries: list[dict[str, object]],
    full_axis: pd.DataFrame,
    edistance: pd.DataFrame,
    artifact_hashes: pd.DataFrame,
) -> str:
    panel_lines = "\n".join(
        f"| {item['panel']} | {item['n_development_cells']} | {item['n_test_cells_metadata_only']} | {item['n_genes']} | "
        f"{item['n_train_conditions']} | {item['n_val_conditions']} | {item['n_test_conditions']} |"
        for item in summaries
    )
    graph_lookup = {item["panel"]: item for item in graph_summaries}
    graph_lines = "\n".join(
        f"- {panel}: {graph_lookup[panel]['n_graph_conditions']} conditions / "
        f"{graph_lookup[panel]['n_cell_graphs']} cell graphs；SHA256 "
        f"`{graph_lookup[panel]['cell_graphs_sha256']}`"
        for panel in graph_lookup
    ) or "- 本次按参数跳过 cell-graph materialization；E157 前仍需完成。"
    return f"""# E156｜PRESCRIBE P3/P4 train-only 预处理报告

## 完成状态

E156 只完成预处理和官方 PertData 兼容资产生成。没有建立或训练模型，没有生成置信度、预测、任务误差、Pearson、方向准确度或 RMSE。

| panel | development cells | test cells（metadata only） | model genes | train conditions | val | test |
|---|---:|---:|---:|---:|---:|---:|
{panel_lines}

## 泄漏控制

HVG、基因检测阈值、PCA 均只在两个面板共享的 64 个训练条件上拟合。模型轴由 train-only top-{N_HVG} HVG 与 E155 已冻结的扰动基因身份取并集，共 {int(full_axis['included_in_model_axis'].sum())} 个基因；没有用 val/test 表达决定基因是否入轴。

每个开发细胞使用固定 `target_sum=10000` 后 `log1p`，没有从 val/test 估计全局归一化常数。PCA10 的均值和 components 只来自 train。训练 E-distance 在 train 内按每条件 {int(edistance['n_balanced'].iloc[0])} 个细胞平衡后计算；val 的 `y_n/y_d/y_s` 是 NaN sentinel。48 个 held-out 测试基因在 train/val 的命中数均为 0，逐基因证明见 `tables/E156_GENE_LEAKAGE_AUDIT.csv`。

E156 为整文件 SHA256 校验读取了原始 H5AD 字节，并以 backed 模式读取 `obs` 元数据；测试 X 行从未被索引、载入内存或执行变换。表中的测试细胞数只来自 E155 已冻结的 `obs`。正式运行时 `perturb_processed.h5ad` 和 `cell_graphs.pkl` 均只有 train+val。E157 必须先锁定 checkpoint 和每任务置信度；随后由 E158 才能读取测试 X，执行固定归一化与 train-PCA transform 并评价。开发资产中的上游 callback 兼容字段使用固定 train-HVG 顺序，不能把它当作 condition-specific top-DE。

两个源码边界已经在训练前登记。第一，native `test_step` 的 `truth` 是 `y_pca` 经 PCA10 逆变换后的重构值，因此 P3/P4 的预注册 Pearson 主终点必须继续用 PCA10-reconstructed truth；raw log-normalized truth只能作敏感性分析。第二，native `ListMLELoss` 对按 `y_n` 排序后的整列做 `-sum(log_softmax)`，该和对排列不变，当前源码中的 E-distance 排序标签实际上不会改变 loss。E156 仍生成 train-only `y_n` 以保持原生接口，但不能宣称它提供了有效的排序监督。逐文件证据见 `tables/E156_NATIVE_CODE_AUDIT.csv`。

## PertData 兼容性

大文件位于 `/home/yyf/data/safeconf_e156_prescribe/`。PRESCRIBE 的 `data/norman_p3` 和 `data/norman_p4` 是指向该目录的符号链接。每个 panel 的运行时 `perturb_processed.h5ad` 和 cell graphs 只含 train+val；E156 不生成测试表达资产。其余资产包括 train-only `perturb_e_distance.h5ad`、冻结 split、PCA prior mean/cov 和训练 E-distance 表。

{graph_lines}

全部 {len(artifact_hashes)} 个计算与数据资产的路径、字节数与 SHA256 在 `tables/E156_ARTIFACT_HASHES.csv`。该表不把脚本、状态、报告和自身计入计算资产。运行脚本 SHA256 与 Git HEAD blob 一致性另存于 `RUN_STATUS.json`。完整 33,694 基因审计、split 输入审计、X/obs provenance 和 train E-distance 均已单独保存。
"""


def main() -> None:
    args = parse_args()
    started = time.time()
    runner_provenance = runner_git_provenance()
    if OUT.exists() or DATA_ROOT.exists():
        raise FileExistsError("E156 output already exists; refusing to overwrite")
    for slug in PANEL_SLUG.values():
        link = PRESCRIBE / "data" / slug
        if link.exists() or link.is_symlink():
            raise FileExistsError(f"PRESCRIBE link already exists: {link}")

    OUT.mkdir(parents=True)
    TABLES.mkdir()
    REPORTS.mkdir()
    DATA_ROOT.mkdir(parents=True)
    write_status(
        experiment="E156_prescribe_norman_p3p4_preprocess",
        phase="started_no_model_training_or_evaluation",
        started_at=datetime.now().isoformat(timespec="seconds"),
        raw_dataset=str(RAW),
        data_root=str(DATA_ROOT),
        **runner_provenance,
        raw_container_opened_for_hash_and_obs=True,
        raw_obs_metadata_read=True,
        test_X_rows_indexed=False,
        test_X_rows_materialized=False,
        test_X_rows_transformed=False,
        model_training_started=False,
        predictions_generated=False,
        test_endpoint_computed=False,
    )
    try:
        e155_status, split_frames, split_dicts = validate_e155()
        write_status(phase="E155_hashes_verified_shared_train_fit_started")
        leakage = leakage_audit(split_dicts)
        leakage.to_csv(TABLES / "E156_GENE_LEAKAGE_AUDIT.csv", index=False)
        full_axis, shared, edistance, balance = build_shared_fit(split_dicts)
        write_shared_assets(full_axis, shared, edistance, balance)
        write_status(
            phase="shared_train_only_fit_complete",
            n_raw_genes=int(len(full_axis)),
            n_model_genes=int(full_axis["included_in_model_axis"].sum()),
            n_train_hvg=int(full_axis["train_seurat_v3_hvg"].sum()),
            n_forced_perturbation_genes=int(full_axis["forced_by_frozen_perturbation_identity"].sum()),
            n_train_edistance_conditions=int(len(edistance)),
            edistance_balance_n=int(edistance["n_balanced"].iloc[0]),
        )

        split_audits = []
        summaries = []
        for panel in PANELS:
            audit, summary = write_panel_assets(
                panel,
                split_frames[panel],
                split_dicts[panel],
                full_axis,
                shared,
                edistance,
            )
            split_audits.append(audit)
            summaries.append(summary)
            write_status(phase=f"{panel}_development_h5ad_complete")
        split_audit = pd.concat(split_audits, ignore_index=True)
        split_audit.to_csv(TABLES / "E156_SPLIT_INPUT_AUDIT.csv", index=False)
        provenance = provenance_table()
        provenance.to_csv(TABLES / "E156_X_OBS_PROVENANCE.csv", index=False)
        code_audit = native_code_audit()
        code_audit.to_csv(TABLES / "E156_NATIVE_CODE_AUDIT.csv", index=False)

        graph_summaries = []
        if not args.skip_cell_graphs:
            for panel in PANELS:
                graph_summaries.append(build_cell_graphs(panel, split_dicts[panel]))
                write_status(phase=f"{panel}_PertData_graphs_complete")

        paths: list[Path] = [
            SHARED / "TRAIN_ONLY_PCA_MODEL.npz",
            SHARED / "train_only_perturb_e_distance.h5ad",
            TABLES / "E156_FULL_GENE_AXIS_AUDIT.csv",
            TABLES / "E156_TRAIN_EDISTANCE.csv",
            TABLES / "E156_BALANCED_TRAIN_CELLS.csv",
            TABLES / "E156_GENE_LEAKAGE_AUDIT.csv",
            TABLES / "E156_SPLIT_INPUT_AUDIT.csv",
            TABLES / "E156_X_OBS_PROVENANCE.csv",
            TABLES / "E156_NATIVE_CODE_AUDIT.csv",
        ]
        for slug in PANEL_SLUG.values():
            panel_dir = DATA_ROOT / slug
            paths.extend(path for path in panel_dir.rglob("*") if path.is_file())
        hashes = artifact_hash_table(paths)
        hashes.to_csv(TABLES / "E156_ARTIFACT_HASHES.csv", index=False)

        report = report_text(summaries, graph_summaries, full_axis, edistance, hashes)
        (REPORTS / "E156_REPORT.md").write_text(report, encoding="utf-8")
        (OUT / "README_先看这个.md").write_text(
            "# E156 先看这个\n\n"
            "先读 `ATTEMPT1_CONTRACT_DEVIATION.md`，再读 `reports/E156_REPORT.md`。"
            "正式 E156 只处理 train+val；第一次提前变换测试 X 的实现已经判为 aborted。"
            "本实验没有训练或评价。\n",
            encoding="utf-8",
        )

        h5ad_hashes = {
            item["panel"]: sha256_file(Path(str(item["h5ad"]))) for item in summaries
        }
        graph_hashes = {
            item["panel"]: item["cell_graphs_sha256"] for item in graph_summaries
        }
        write_status(
            phase="complete_preprocessing_only_no_training_no_evaluation",
            finished_at=datetime.now().isoformat(timespec="seconds"),
            runtime_seconds=round(time.time() - started, 3),
            e155_phase=e155_status["phase"],
            e155_commit_prefix=EXPECTED_E155_COMMIT,
            e155_artifact_hashes_verified=True,
            raw_sha256=EXPECTED_RAW_SHA256,
            normalization="fixed per-cell target_sum=10000 then log1p",
            hvg_fit_scope="shared E155 train only",
            pca_fit_scope="shared E155 train only",
            edistance_fit_scope="shared E155 train only",
            validation_test_label_policy=(
                "validation uses NaN sentinels; raw container/obs inspected but test-X rows "
                "were not indexed, materialized, or transformed"
            ),
            native_rank_label_effective=False,
            native_rank_label_reason="ListMLE -sum(log_softmax) is invariant to y_n-induced permutation",
            primary_truth_definition="native PCA10-reconstructed truth, matching E95/E145; raw normalized truth sensitivity only",
            native_loss_sha256=sha256_file(PRESCRIBE / "src/nn/loss.py"),
            native_lightning_module_sha256=sha256_file(PRESCRIBE / "src/model/lightening_module.py"),
            test_expression_role=(
                "raw container and obs metadata inspected; test-X rows never indexed, "
                "materialized, or transformed; E155 obs metadata counts only"
            ),
            test_expression_opened=False,
            test_expression_opened_definition="no test-X row indexing/materialization; raw container and obs were opened",
            test_expression_transformed=False,
            raw_container_opened_for_hash_and_obs=True,
            raw_obs_metadata_read=True,
            test_X_rows_indexed=False,
            test_X_rows_materialized=False,
            test_X_rows_transformed=False,
            model_training_started=False,
            predictions_generated=False,
            test_endpoint_computed=False,
            heldout_gene_development_leakage_count=int(leakage["development_leakage"].sum()),
            panels=summaries,
            graph_summaries=graph_summaries,
            perturb_processed_sha256=h5ad_hashes,
            cell_graphs_sha256=graph_hashes,
            artifact_manifest_sha256=sha256_file(TABLES / "E156_ARTIFACT_HASHES.csv"),
        )
        print((OUT / "RUN_STATUS.json").read_text(encoding="utf-8"))
    except Exception as exc:
        write_status(
            phase="failed_preprocessing_no_training_no_evaluation",
            failed_at=datetime.now().isoformat(timespec="seconds"),
            error=repr(exc),
            traceback=traceback.format_exc(),
            model_training_started=False,
            predictions_generated=False,
            test_endpoint_computed=False,
        )
        raise


if __name__ == "__main__":
    main()
