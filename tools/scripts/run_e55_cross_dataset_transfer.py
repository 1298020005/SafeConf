#!/usr/bin/env python3
"""E55 cross-dataset transfer audit.

This experiment implements the hardest setting that the advisor explicitly
asked for: use one dataset as the historical/source domain and score prediction
risk on another dataset as the target domain.

The score is deliberately deployable:

* source perturbation support;
* source/target control-state similarity;
* disagreement between two source-only reference predictors;
* predicted effect magnitude.

The held-out target truth is used only after scoring, to compute prediction
error.  True effect magnitude is kept as an oracle diagnostic, not as an input.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[2]
ATLAS = Path("/home/yyf/data/singlecell_perturbation_atlas")
OUT = ROOT / "docs" / "实验结果" / "E55_cross_dataset_transfer_20260710"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"

CONTROL_VALUES = {
    "",
    "control",
    "ctrl",
    "vehicle",
    "dmso",
    "nt",
    "ntc",
    "negctrl",
    "non-targeting",
    "non_targeting",
    "intergenic",
    "nan",
    "none",
    "null",
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: Path
    context_col: str
    perturbation_col: str
    control_col: str | None = None
    context_normalizer: str | None = None
    family: str = ""
    note: str = ""


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        return bool(subprocess.check_output(["git", "status", "--short"], cwd=ROOT).decode().strip())
    except Exception:
        return True


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except Exception:
        try:
            return path.relative_to(ATLAS).as_posix()
        except Exception:
            return str(path)


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)


def as_dense(x) -> np.ndarray:
    if sp.issparse(x):
        return x.toarray()
    return np.asarray(x)


def normalize_token(x: object) -> str:
    text = "" if pd.isna(x) else str(x).strip().lower()
    text = text.replace("+", "plus")
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def normalize_context(x: object, mode: str | None) -> str:
    text = "" if pd.isna(x) else str(x).strip()
    if mode == "kang_cell_type":
        text = text.replace("+", "")
        text = re.sub(r"[\s_]+", "_", text)
        text = text.replace("CD14_Mono", "CD14_Mono")
        text = text.replace("CD16_Mono", "CD16_Mono")
        text = text.replace("CD4_T_cells", "CD4_T")
        text = text.replace("CD4_T", "CD4_T")
        text = text.replace("CD8_T_cells", "CD8_T")
        text = text.replace("CD8_T", "CD8_T")
        text = text.replace("B_cells", "B")
        text = text.replace("Dendritic_cells", "DC")
        text = text.replace("FCGR3A_Monocytes", "CD16_Mono")
        text = text.replace("NK_cells", "NK")
    return text


def is_control_value(x: object) -> bool:
    text = "" if pd.isna(x) else str(x).strip()
    low = text.lower()
    token = normalize_token(text)
    return (
        text in CONTROL_VALUES
        or low in CONTROL_VALUES
        or token in CONTROL_VALUES
        or low.startswith("control")
        or low.startswith("ctrl")
        or token.startswith("negctrl")
        or token.startswith("intergenic")
        or "negativecontrol" in token
    )


def is_control_group(obs: pd.DataFrame, perturbation_col: str, control_col: str | None) -> pd.Series:
    by_pert = obs[perturbation_col].map(is_control_value)
    if control_col is None or control_col not in obs.columns:
        return by_pert.astype(bool)
    raw = obs[control_col]
    if pd.api.types.is_bool_dtype(raw):
        by_flag = raw.fillna(False).astype(bool)
    else:
        low = raw.astype(str).str.lower().str.strip()
        by_flag = low.isin({"true", "1", "yes", "control", "ctrl"})
    return (by_pert | by_flag).astype(bool)


def choose_common_genes(source: ad.AnnData, target: ad.AnnData, n_genes: int) -> list[str]:
    target_genes = set(map(str, target.var_names))
    common = [str(g) for g in source.var_names if str(g) in target_genes]
    if len(common) <= n_genes:
        return common

    source_hv = None
    target_hv = None
    for col in source.var.columns:
        if str(col).startswith("highly_variable"):
            source_hv = set(map(str, source.var_names[np.asarray(source.var[col]).astype(bool)]))
    for col in target.var.columns:
        if str(col).startswith("highly_variable"):
            target_hv = set(map(str, target.var_names[np.asarray(target.var[col]).astype(bool)]))
    if source_hv is not None and target_hv is not None:
        hv_common = [g for g in common if g in source_hv and g in target_hv]
        if len(hv_common) >= max(50, min(n_genes, 200)):
            return hv_common[:n_genes]
    if source_hv is not None:
        hv_common = [g for g in common if g in source_hv]
        if len(hv_common) >= max(50, min(n_genes, 200)):
            return hv_common[:n_genes]
    return common[:n_genes]


def get_matrix(adata: ad.AnnData, genes: list[str]) -> np.ndarray:
    layer = "logNor" if "logNor" in adata.layers else ("logcounts" if "logcounts" in adata.layers else None)
    view = adata[:, genes]
    x = view.layers[layer] if layer else view.X
    return as_dense(x).astype(np.float32)


def sample_positions(pos: np.ndarray, max_cells: int, rng: np.random.Generator) -> np.ndarray:
    if len(pos) > max_cells:
        return np.asarray(rng.choice(pos, size=max_cells, replace=False), dtype=int)
    return np.asarray(pos, dtype=int)


def build_tasks_for_genes(
    spec: DatasetSpec,
    genes: list[str],
    min_cells: int,
    max_cells_per_group: int,
    seed: int,
) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(seed)
    adata = ad.read_h5ad(spec.path)
    obs = adata.obs.copy()
    global_context = spec.context_col in {"__global__", "global"} and spec.context_col not in obs.columns
    missing = [c for c in [spec.perturbation_col] if c not in obs.columns]
    if not global_context and spec.context_col not in obs.columns:
        missing.append(spec.context_col)
    if missing:
        raise KeyError(f"{spec.name} missing obs columns: {missing}")
    x = get_matrix(adata, genes)
    if global_context:
        obs["_context"] = "global"
    else:
        obs["_context"] = obs[spec.context_col].map(lambda v: normalize_context(v, spec.context_normalizer))
    obs["_perturbation"] = obs[spec.perturbation_col].astype(str)
    obs["_perturbation_key"] = obs["_perturbation"].map(normalize_token)
    obs["_is_control"] = is_control_group(obs, spec.perturbation_col, spec.control_col)

    group_pos: dict[tuple[str, str], np.ndarray] = {}
    group_key: dict[tuple[str, str], str] = {}
    group_is_control: dict[tuple[str, str], bool] = {}
    for (ctx, pert), sub in obs.groupby(["_context", "_perturbation"], observed=False):
        pos = obs.index.get_indexer(sub.index)
        if len(pos) < min_cells:
            continue
        key = str(sub["_perturbation_key"].mode().iloc[0])
        group_pos[(str(ctx), str(pert))] = sample_positions(pos, max_cells_per_group, rng)
        group_key[(str(ctx), str(pert))] = key
        group_is_control[(str(ctx), str(pert))] = bool(sub["_is_control"].any() or is_control_value(pert))

    control_means: dict[str, np.ndarray] = {}
    control_cells: dict[str, int] = {}
    for (ctx, pert), pos in group_pos.items():
        if group_is_control.get((ctx, pert), False):
            control_means[ctx] = x[pos].mean(axis=0).astype(np.float32)
            control_cells[ctx] = int(len(pos))

    tasks = []
    for (ctx, pert), pos in group_pos.items():
        if group_is_control.get((ctx, pert), False):
            continue
        if ctx not in control_means:
            continue
        effect = x[pos].mean(axis=0) - control_means[ctx]
        tasks.append(
            {
                "dataset": spec.name,
                "family": spec.family,
                "context": str(ctx),
                "perturbation": str(pert),
                "perturbation_key": group_key[(ctx, pert)],
                "effect": effect.astype(np.float32),
                "control_mean": control_means[ctx].astype(np.float32),
                "n_cells": int(len(pos)),
                "n_control_cells": int(control_cells.get(ctx, 0)),
            }
        )

    meta = {
        "dataset_name": spec.name,
        "path": rel(spec.path),
        "family": spec.family,
        "note": spec.note,
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "n_genes_used": int(len(genes)),
        "context_col": spec.context_col,
        "perturbation_col": spec.perturbation_col,
        "control_col": spec.control_col or "",
        "n_tasks": int(len(tasks)),
        "n_contexts": int(len({t["context"] for t in tasks})),
        "n_perturbations": int(len({t["perturbation_key"] for t in tasks})),
        "n_control_contexts": int(len(control_means)),
    }
    return tasks, meta


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    den = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if den <= 1e-12:
        return 0.0
    return float(np.dot(aa, bb) / den)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) ** 2)))


def vec_l2(a: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float)))


def zscore(s: pd.Series) -> pd.Series:
    arr = pd.to_numeric(s, errors="coerce")
    sd = float(arr.std(ddof=0))
    if not np.isfinite(sd) or sd <= 1e-12:
        return pd.Series(np.zeros(len(arr)), index=s.index, dtype=float)
    return (arr - float(arr.mean())) / sd


def spearman(x: pd.Series, y: pd.Series) -> float:
    df = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < 3 or df["x"].nunique() < 2 or df["y"].nunique() < 2:
        return float("nan")
    return float(df["x"].corr(df["y"], method="spearman"))


def top_enrichment(df: pd.DataFrame, score_col: str, error_col: str, frac: float = 0.2) -> tuple[int, float, float]:
    sub = df[[score_col, error_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 5:
        return 0, float("nan"), float("nan")
    k = max(1, int(math.ceil(len(sub) * frac)))
    top = sub.sort_values(score_col, ascending=False).head(k)
    all_mean = float(sub[error_col].mean())
    top_mean = float(top[error_col].mean())
    return k, top_mean, top_mean / all_mean if all_mean > 1e-12 else float("nan")


def source_indices(tasks: list[dict]) -> dict[str, object]:
    effects = np.stack([t["effect"] for t in tasks]).astype(np.float32)
    global_mean = effects.mean(axis=0).astype(np.float32)

    by_pert: dict[str, list[int]] = {}
    by_context: dict[str, list[int]] = {}
    context_control: dict[str, np.ndarray] = {}
    for i, t in enumerate(tasks):
        by_pert.setdefault(t["perturbation_key"], []).append(i)
        by_context.setdefault(t["context"], []).append(i)
        context_control.setdefault(t["context"], t["control_mean"])

    pert_mean = {k: effects[idx].mean(axis=0).astype(np.float32) for k, idx in by_pert.items()}
    context_mean = {k: effects[idx].mean(axis=0).astype(np.float32) for k, idx in by_context.items()}

    return {
        "effects": effects,
        "global_mean": global_mean,
        "by_pert": by_pert,
        "by_context": by_context,
        "context_control": context_control,
        "pert_mean": pert_mean,
        "context_mean": context_mean,
    }


def score_pair(
    source_spec: DatasetSpec,
    target_spec: DatasetSpec,
    source_tasks: list[dict],
    target_tasks: list[dict],
    n_common_genes: int,
    pair_group: str,
) -> pd.DataFrame:
    idx = source_indices(source_tasks)
    source_contexts = sorted(idx["context_control"].keys())

    rows = []
    for pos, task in enumerate(target_tasks):
        pert_key = task["perturbation_key"]
        support_ids = idx["by_pert"].get(pert_key, [])
        support_count = len(support_ids)

        p_pert = idx["pert_mean"].get(pert_key, idx["global_mean"])

        sims = [(ctx, cosine(task["control_mean"], idx["context_control"][ctx])) for ctx in source_contexts]
        nearest_ctx, nearest_sim = max(sims, key=lambda x: x[1]) if sims else ("", 0.0)

        if support_ids:
            best_id = max(support_ids, key=lambda i: cosine(task["control_mean"], source_tasks[i]["control_mean"]))
            p_ctx = source_tasks[best_id]["effect"]
            context_predictor_mode = "nearest_context_same_perturbation"
        elif nearest_ctx in idx["context_mean"]:
            p_ctx = idx["context_mean"][nearest_ctx]
            context_predictor_mode = "nearest_context_mean_effect"
        else:
            p_ctx = idx["global_mean"]
            context_predictor_mode = "global_fallback"

        p_combined = 0.5 * (np.asarray(p_pert) + np.asarray(p_ctx))
        true = task["effect"]
        rows.append(
            {
                "pair_group": pair_group,
                "source_dataset": source_spec.name,
                "target_dataset": target_spec.name,
                "directional_pair": f"{source_spec.name} -> {target_spec.name}",
                "target_task_index": int(pos),
                "target_context": task["context"],
                "target_perturbation": task["perturbation"],
                "target_perturbation_key": pert_key,
                "source_family": source_spec.family,
                "target_family": target_spec.family,
                "n_common_genes": int(n_common_genes),
                "source_support_count": int(support_count),
                "source_n_tasks": int(len(source_tasks)),
                "source_n_contexts": int(len(source_contexts)),
                "target_n_cells": int(task["n_cells"]),
                "target_n_control_cells": int(task["n_control_cells"]),
                "nearest_source_context": nearest_ctx,
                "nearest_context_similarity": float(nearest_sim),
                "context_predictor_mode": context_predictor_mode,
                "prediction_disagreement_rmse": rmse(p_pert, p_ctx),
                "predicted_l2_combined": vec_l2(p_combined),
                "predicted_l2_perturbation_mean": vec_l2(p_pert),
                "predicted_l2_context": vec_l2(p_ctx),
                "error_perturbation_mean_rmse": rmse(p_pert, true),
                "error_context_predictor_rmse": rmse(p_ctx, true),
                "error_combined_rmse": rmse(p_combined, true),
                "true_l2_diagnostic": vec_l2(true),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["risk_low_source_support"] = -zscore(np.log1p(out["source_support_count"]))
    out["risk_low_context_similarity"] = -zscore(out["nearest_context_similarity"])
    out["risk_disagreement"] = zscore(out["prediction_disagreement_rmse"])
    out["risk_predicted_magnitude"] = zscore(out["predicted_l2_combined"])
    out["risk_cross_dataset"] = (
        out["risk_low_source_support"]
        + out["risk_low_context_similarity"]
        + out["risk_disagreement"]
        + out["risk_predicted_magnitude"]
    )
    out["risk_oracle_true_magnitude_diagnostic"] = zscore(out["true_l2_diagnostic"])
    return out


def summarize_scores(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if scores.empty:
        return pd.DataFrame()
    score_cols = [
        "risk_cross_dataset",
        "risk_disagreement",
        "risk_predicted_magnitude",
        "risk_low_source_support",
        "risk_low_context_similarity",
        "risk_oracle_true_magnitude_diagnostic",
    ]
    for (pair_group, src, tgt), sub in scores.groupby(["pair_group", "source_dataset", "target_dataset"], observed=False):
        base = {
            "pair_group": pair_group,
            "source_dataset": src,
            "target_dataset": tgt,
            "directional_pair": f"{src} -> {tgt}",
            "n_target_tasks": int(len(sub)),
            "n_common_genes": int(sub["n_common_genes"].iloc[0]) if len(sub) else 0,
            "shared_perturbation_tasks": int((sub["source_support_count"] > 0).sum()),
            "mean_error_combined_rmse": float(sub["error_combined_rmse"].mean()),
            "mean_nearest_context_similarity": float(sub["nearest_context_similarity"].mean()),
        }
        for col in score_cols:
            k, top_mean, enrich = top_enrichment(sub, col, "error_combined_rmse", frac=0.2)
            rows.append(
                {
                    **base,
                    "score_col": col,
                    "spearman_vs_error": spearman(sub[col], sub["error_combined_rmse"]),
                    "top20_k": int(k),
                    "top20_mean_error": top_mean,
                    "top20_error_enrichment": enrich,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["pair_group", "source_dataset", "target_dataset", "score_col"]).reset_index(drop=True)


def pair_status_row(
    pair_group: str,
    source_spec: DatasetSpec,
    target_spec: DatasetSpec,
    status: str,
    message: str,
    n_common_genes: int = 0,
    source_meta: dict | None = None,
    target_meta: dict | None = None,
) -> dict:
    source_meta = source_meta or {}
    target_meta = target_meta or {}
    return {
        "pair_group": pair_group,
        "source_dataset": source_spec.name,
        "target_dataset": target_spec.name,
        "directional_pair": f"{source_spec.name} -> {target_spec.name}",
        "status": status,
        "message": message,
        "n_common_genes": int(n_common_genes),
        "source_n_tasks": int(source_meta.get("n_tasks", 0) or 0),
        "target_n_tasks": int(target_meta.get("n_tasks", 0) or 0),
        "source_n_contexts": int(source_meta.get("n_contexts", 0) or 0),
        "target_n_contexts": int(target_meta.get("n_contexts", 0) or 0),
        "source_n_perturbations": int(source_meta.get("n_perturbations", 0) or 0),
        "target_n_perturbations": int(target_meta.get("n_perturbations", 0) or 0),
        "source_path": rel(source_spec.path),
        "target_path": rel(target_spec.path),
    }


def build_specs() -> dict[str, DatasetSpec]:
    base = ATLAS / "extra_official" / "cellular_context_generalization"
    return {
        "KaggleCrossCell_celltype": DatasetSpec(
            name="KaggleCrossCell_celltype",
            path=base / "KaggleCrossCell.h5ad",
            context_col="cell_type",
            perturbation_col="condition2",
            control_col="control",
            family="chemical_pbmc",
            note="OpenProblems-style PBMC drug panel; context is cell type.",
        ),
        "KaggleCrossPatient_celltype": DatasetSpec(
            name="KaggleCrossPatient_celltype",
            path=base / "KaggleCrossPatient.h5ad",
            context_col="cell_type",
            perturbation_col="condition2",
            control_col="control",
            family="chemical_pbmc",
            note="Same drug panel, different file; context is cell type.",
        ),
        "KaggleCrossPatient_donor": DatasetSpec(
            name="KaggleCrossPatient_donor",
            path=base / "KaggleCrossPatient.h5ad",
            context_col="donor_id",
            perturbation_col="condition2",
            control_col="control",
            family="chemical_pbmc",
            note="Same file as KaggleCrossPatient, but context is donor/patient.",
        ),
        "sciplex3_cellline": DatasetSpec(
            name="sciplex3_cellline",
            path=base / "sciplex3.h5ad",
            context_col="condition1",
            perturbation_col="condition2",
            family="chemical_cellline",
            note="SciPlex3 processed subset; context is cell line, perturbation is drug identity collapsed over dose.",
        ),
        "McFarland_cellline": DatasetSpec(
            name="McFarland_cellline",
            path=base / "McFarland.h5ad",
            context_col="condition1",
            perturbation_col="condition2",
            family="chemical_cellline",
            note="Cancer cell-line drug perturbation panel.",
        ),
        "crossPatient_patient": DatasetSpec(
            name="crossPatient_patient",
            path=base / "crossPatient.h5ad",
            context_col="condition1",
            perturbation_col="condition2",
            family="chemical_patient",
            note="Patient-level drug response panel.",
        ),
        "kangCrossCell_celltype": DatasetSpec(
            name="kangCrossCell_celltype",
            path=base / "kangCrossCell.h5ad",
            context_col="cell_type",
            perturbation_col="condition2",
            family="immune_stim",
            note="Kang IFN stimulation; context is cell type.",
        ),
        "kangCrossPatient_celltype": DatasetSpec(
            name="kangCrossPatient_celltype",
            path=base / "kangCrossPatient.h5ad",
            context_col="cell_type_x",
            perturbation_col="condition2",
            context_normalizer="kang_cell_type",
            family="immune_stim",
            note="Kang IFN stimulation second file; context is normalized cell type.",
        ),
        "TCDD_mouse_liver": DatasetSpec(
            name="TCDD_mouse_liver",
            path=base / "TCDD.h5ad",
            context_col="condition1",
            perturbation_col="condition2",
            family="chemical_mouse_liver",
            note="Mouse liver dose-response; included for feasibility audit, not forced into human chemical transfer.",
        ),
    }


def planned_pairs(specs: dict[str, DatasetSpec]) -> list[tuple[str, DatasetSpec, DatasetSpec]]:
    pairs: list[tuple[str, DatasetSpec, DatasetSpec]] = []
    same_system = [
        ("KaggleCrossCell_celltype", "KaggleCrossPatient_celltype"),
        ("KaggleCrossPatient_celltype", "KaggleCrossCell_celltype"),
        ("KaggleCrossCell_celltype", "KaggleCrossPatient_donor"),
        ("KaggleCrossPatient_donor", "KaggleCrossCell_celltype"),
        ("kangCrossCell_celltype", "kangCrossPatient_celltype"),
        ("kangCrossPatient_celltype", "kangCrossCell_celltype"),
    ]
    for a, b in same_system:
        pairs.append(("same_system_cross_file", specs[a], specs[b]))

    chemical = [
        "KaggleCrossCell_celltype",
        "KaggleCrossPatient_celltype",
        "sciplex3_cellline",
        "McFarland_cellline",
        "crossPatient_patient",
    ]
    for a, b in itertools.permutations(chemical, 2):
        # The Kaggle pair is already covered above with a clearer label.
        if {a, b} == {"KaggleCrossCell_celltype", "KaggleCrossPatient_celltype"}:
            continue
        pairs.append(("hard_chemical_cross_dataset", specs[a], specs[b]))

    # Feasibility checks that should not be hidden.  These answer "can this be
    # computed in this setting?" without pretending every pair is a good design.
    pairs.extend(
        [
            ("feasibility_boundary", specs["sciplex3_cellline"], specs["TCDD_mouse_liver"]),
            ("feasibility_boundary", specs["TCDD_mouse_liver"], specs["sciplex3_cellline"]),
        ]
    )
    return pairs


def run(args: argparse.Namespace) -> None:
    ensure_dirs()
    specs = build_specs()
    pairs = planned_pairs(specs)
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]

    all_scores: list[pd.DataFrame] = []
    pair_rows: list[dict] = []
    dataset_rows: list[dict] = []

    for i, (pair_group, source_spec, target_spec) in enumerate(pairs, start=1):
        print(f"[E55] {i}/{len(pairs)} {pair_group}: {source_spec.name} -> {target_spec.name}", flush=True)
        try:
            source_head = ad.read_h5ad(source_spec.path, backed="r")
            target_head = ad.read_h5ad(target_spec.path, backed="r")
            genes = choose_common_genes(source_head, target_head, args.n_genes)
            n_common = len(genes)
            if n_common < args.min_common_genes:
                msg = f"too few common genes: {n_common} < {args.min_common_genes}"
                pair_rows.append(pair_status_row(pair_group, source_spec, target_spec, "skipped_too_few_common_genes", msg, n_common))
                print(f"  - skip: {msg}", flush=True)
                continue

            source_tasks, source_meta = build_tasks_for_genes(
                source_spec,
                genes,
                min_cells=args.min_cells,
                max_cells_per_group=args.max_cells_per_group,
                seed=args.seed,
            )
            target_tasks, target_meta = build_tasks_for_genes(
                target_spec,
                genes,
                min_cells=args.min_cells,
                max_cells_per_group=args.max_cells_per_group,
                seed=args.seed + 17,
            )
            dataset_rows.extend([{**source_meta, "role": "source", "pair": f"{source_spec.name}->{target_spec.name}"}, {**target_meta, "role": "target", "pair": f"{source_spec.name}->{target_spec.name}"}])

            if len(source_tasks) < args.min_source_tasks or len(target_tasks) < args.min_target_tasks:
                msg = f"too few tasks: source={len(source_tasks)}, target={len(target_tasks)}"
                pair_rows.append(pair_status_row(pair_group, source_spec, target_spec, "skipped_too_few_tasks", msg, n_common, source_meta, target_meta))
                print(f"  - skip: {msg}", flush=True)
                continue

            score = score_pair(source_spec, target_spec, source_tasks, target_tasks, n_common, pair_group)
            if score.empty:
                msg = "score table is empty"
                pair_rows.append(pair_status_row(pair_group, source_spec, target_spec, "empty_score", msg, n_common, source_meta, target_meta))
                print(f"  - skip: {msg}", flush=True)
                continue
            all_scores.append(score)
            pair_rows.append(pair_status_row(pair_group, source_spec, target_spec, "ok", "scored", n_common, source_meta, target_meta))
            print(f"  - ok: source_tasks={len(source_tasks)}, target_tasks={len(target_tasks)}, genes={n_common}", flush=True)
        except Exception as exc:
            pair_rows.append(pair_status_row(pair_group, source_spec, target_spec, "failed", repr(exc)))
            print(f"  - failed: {exc!r}", flush=True)

    scores = pd.concat(all_scores, ignore_index=True) if all_scores else pd.DataFrame()
    summary = summarize_scores(scores)
    pair_status = pd.DataFrame(pair_rows)
    dataset_status = pd.DataFrame(dataset_rows).drop_duplicates().reset_index(drop=True) if dataset_rows else pd.DataFrame()

    scores.to_csv(TABLES / "E55_CROSS_DATASET_SCORE_TABLE.csv", index=False)
    summary.to_csv(TABLES / "E55_CROSS_DATASET_SUMMARY.csv", index=False)
    pair_status.to_csv(TABLES / "E55_PAIR_STATUS.csv", index=False)
    dataset_status.to_csv(TABLES / "E55_DATASET_TASK_STATUS.csv", index=False)

    run_status = {
        "experiment": "E55_cross_dataset_transfer",
        "created_at": now_text(),
        "git_head": git_head(),
        "git_dirty": git_dirty(),
        "args": vars(args),
        "output_dir": rel(OUT),
        "n_pairs_planned": len(pairs),
        "n_pairs_ok": int((pair_status["status"] == "ok").sum()) if not pair_status.empty else 0,
        "n_score_rows": int(len(scores)),
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(run_status, ensure_ascii=False, indent=2), encoding="utf-8")
    write_reports(scores, summary, pair_status, dataset_status, run_status)


def fmt(x: object, digits: int = 3) -> str:
    try:
        v = float(x)
    except Exception:
        return ""
    if not np.isfinite(v):
        return "nan"
    return f"{v:.{digits}f}"


def write_reports(
    scores: pd.DataFrame,
    summary: pd.DataFrame,
    pair_status: pd.DataFrame,
    dataset_status: pd.DataFrame,
    run_status: dict,
) -> None:
    ok_pairs = pair_status[pair_status["status"] == "ok"].copy() if not pair_status.empty else pd.DataFrame()
    main = summary[summary["score_col"] == "risk_cross_dataset"].copy() if not summary.empty else pd.DataFrame()
    if not main.empty:
        main = main.sort_values(["pair_group", "spearman_vs_error"], ascending=[True, False])

    lines = [
        "# E55 跨数据集 transfer 审计",
        "",
        "这一轮专门回答老师最后强调的 setting：一个数据集作为历史经验，另一个数据集作为目标场景。",
        "",
        "分数输入只包含预测前能拿到的东西：源数据支持数、源/目标 control 状态相似度、两个源域参考预测器的分歧、预测效应大小。目标真值只用于最后计算误差。",
        "",
        "## 1. 跑了什么",
        "",
        f"- 计划方向对：{run_status['n_pairs_planned']}",
        f"- 成功打分方向对：{run_status['n_pairs_ok']}",
        f"- 目标任务打分行数：{run_status['n_score_rows']}",
        "",
        "数据分两类：",
        "",
        "- same_system_cross_file：同一研究体系的跨文件迁移，例如 KaggleCrossCell 到 KaggleCrossPatient、Kang CrossCell 到 CrossPatient。",
        "- hard_chemical_cross_dataset：不同化学扰动数据集互相迁移，例如 Kaggle、SciPlex3、McFarland、crossPatient。",
        "- feasibility_boundary：只做可计算性检查。比如 sciplex3 和 TCDD 基因交集太少，不能硬做结论。",
        "",
        "## 2. 主结果表",
        "",
    ]
    if main.empty:
        lines.append("暂无成功 summary。")
    else:
        lines.extend(
            [
                "| 分组 | 方向 | 任务数 | 共同基因 | 共享扰动任务 | ρ(risk,error) | top20 错误富集 | 平均误差 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for _, r in main.iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(r["pair_group"]),
                        str(r["directional_pair"]),
                        str(int(r["n_target_tasks"])),
                        str(int(r["n_common_genes"])),
                        str(int(r["shared_perturbation_tasks"])),
                        fmt(r["spearman_vs_error"]),
                        fmt(r["top20_error_enrichment"]),
                        fmt(r["mean_error_combined_rmse"]),
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## 3. 怎么给老师解释",
            "",
            "汇报时可以这样说：",
            "",
            "> 老师上次提到跨数据集预测，我这次把它单独拆出来做了。源数据集只提供历史支持、control 状态和参考预测器，目标数据集的真实效应没有进入打分。最后再用目标真值算误差，看这个风险分数能不能把更容易错的任务排到前面。",
            "",
            "如果 same_system_cross_file 为正，说明在相同研究体系内换一个文件/划分，风险排序还有迁移性。",
            "",
            "hard_chemical_cross_dataset 如果很弱，就按边界处理：不同药物面板、不同细胞体系、共同基因较少时，源域经验未必能直接迁移。这部分写进限制和下一步，别硬吹。",
            "",
            "## 4. 和老师要求逐条对应",
            "",
            "| 老师的要求 | 当前对应证据 | 状态 |",
            "|---|---|---|",
            "| 分数到底和谁的误差相关 | E33 已审计：误差来自参考预测器；E55 继续用 source-only 预测器并记录 error_combined_rmse | 已补 |",
            "| 输入不能偷看测试答案 | E55 的 risk_cross_dataset 不含 true_l2；true_l2 只标成 oracle diagnostic | 已补 |",
            "| 小矩阵/低覆盖 | E34/E35 split smoke，E49-E52 正式化里已有低支持和留出版本 | 已有 |",
            "| 整行/整列留出 | E34/E35、E49/E50/E52 覆盖 leave-context / leave-perturbation / dose-aware | 已有 |",
            "| 一个数据集到另一个数据集 | 本轮 E55 | 新增 |",
            "| 不同数据类型 | E40-E54 覆盖 chemical、gene combo、dose、regulatory、多模态审计；E55 重点补 chemical/immune 跨数据集 | 持续补 |",
            "",
            "## 5. 文件",
            "",
            f"- 分数明细：`{rel(TABLES / 'E55_CROSS_DATASET_SCORE_TABLE.csv')}`",
            f"- 汇总表：`{rel(TABLES / 'E55_CROSS_DATASET_SUMMARY.csv')}`",
            f"- pair 状态：`{rel(TABLES / 'E55_PAIR_STATUS.csv')}`",
            f"- 数据任务状态：`{rel(TABLES / 'E55_DATASET_TASK_STATUS.csv')}`",
            f"- 运行状态：`{rel(OUT / 'RUN_STATUS.json')}`",
        ]
    )
    (REPORTS / "E55_CROSS_DATASET_TRANSFER_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    readme = [
        "# E55 先看这个",
        "",
        "本目录是跨数据集 transfer 审计。它回答老师说的：在一个数据集见过，换另一个数据集预测时，SafeConf 这类风险排序还站不站得住。",
        "",
        "建议阅读顺序：",
        "",
        "1. `reports/E55_CROSS_DATASET_TRANSFER_REPORT.md`",
        "2. `tables/E55_CROSS_DATASET_SUMMARY.csv`",
        "3. `tables/E55_PAIR_STATUS.csv`",
        "4. `tables/E55_CROSS_DATASET_SCORE_TABLE.csv`",
        "",
        "注意：`risk_oracle_true_magnitude_diagnostic` 只做诊断，不进入可部署打分。",
    ]
    (OUT / "README_先看这个.md").write_text("\n".join(readme) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-genes", type=int, default=1000)
    p.add_argument("--min-common-genes", type=int, default=100)
    p.add_argument("--min-cells", type=int, default=20)
    p.add_argument("--max-cells-per-group", type=int, default=400)
    p.add_argument("--min-source-tasks", type=int, default=3)
    p.add_argument("--min-target-tasks", type=int, default=3)
    p.add_argument("--seed", type=int, default=55)
    p.add_argument("--max-pairs", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
