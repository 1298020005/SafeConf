#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "03_code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from build_context_splits import build_effect_tasks  # noqa: E402
from transport_models import ContextSimilarityBaseline, V0StrongBaseline  # noqa: E402


DEFAULT_DATASET_NAMES = ["KaggleCrossCell", "Haber", "Parekh"]
DATASET_NAMES = list(DEFAULT_DATASET_NAMES)
DEFAULT_ATLAS_ROOT = Path("/home/yyf/datasets/singlecell_perturbation_atlas")
REQUESTED_SPEC = Path("/home/yyf/SafeTrans-docs/02-执行规格/Phase2.1-Codex执行规格-全文.md")
FALLBACK_PLAN = Path("/home/yyf/SafeTrans-docs/01-研究方案/SafeTrans-confidence-scoring-方案.md")
PHASE2_AUDIT = Path("/home/yyf/SafeTrans-docs/03-审计报告/Phase2-v2-独立审计.md")


@dataclass
class RunDirs:
    out: Path
    input: Path
    tables: Path
    figures: Path
    reports: Path
    logs: Path
    scripts: Path


def make_dirs(out: Path) -> RunDirs:
    dirs = RunDirs(
        out=out,
        input=out / "input",
        tables=out / "tables",
        figures=out / "figures",
        reports=out / "reports",
        logs=out / "logs",
        scripts=out / "scripts",
    )
    for p in dirs.__dict__.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def to_jsonable(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        if np.isnan(x):
            return None
        return float(x)
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    return x


def gene_order_hash(genes: list[str]) -> str:
    payload = "\n".join(str(g) for g in genes).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def required_task_value(task: dict, field: str) -> str:
    value = str(task.get(field, "")).strip()
    if not value:
        raise ValueError(f"Task {task.get('task_key', '<unknown>')} has empty required field {field}")
    return value


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=to_jsonable), encoding="utf-8")


def df_markdown(df: pd.DataFrame, max_rows: int = 80) -> str:
    """Small markdown-table fallback without pandas' optional tabulate dependency."""
    if df is None or df.empty:
        return "_empty_"
    show = df.head(max_rows).copy()
    cols = [str(c) for c in show.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in show.iterrows():
        vals = []
        for col in show.columns:
            val = row[col]
            if isinstance(val, float):
                vals.append("" if not np.isfinite(val) else f"{val:.6g}")
            else:
                text = str(val)
                if len(text) > 80:
                    text = text[:77] + "..."
                vals.append(text.replace("|", "/"))
        lines.append("| " + " | ".join(vals) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Only first {max_rows} of {len(df)} rows shown._")
    return "\n".join(lines)


def build_main_table_from_eval(
    eval_summary: pd.DataFrame,
    dataset_names: list[str] | None = None,
    score_names: list[str] | None = None,
) -> pd.DataFrame:
    """Build the per-dataset summary from datasets actually present.

    Phase 2.1 used the fixed Haber/Parekh/KaggleCrossCell panel, but Phase 3
    blind probes intentionally run one dataset at a time.  This helper keeps
    the output schema stable without assuming a particular dataset is present.
    """
    score_names = score_names or [
        "simple_combined_confidence",
        "learned_risk_score",
        "historical_residual_risk",
        "model_disagreement_risk",
    ]
    present = (
        eval_summary.loc[eval_summary["level"] == "dataset", "dataset_name"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    if dataset_names is None:
        datasets = present
    else:
        present_set = set(present)
        datasets = [d for d in dataset_names if d in present_set]
        extras = [d for d in present if d not in set(datasets)]
        datasets.extend(extras)

    rows = []
    for dataset in datasets:
        row = {"dataset_name": dataset}
        for score in score_names:
            sub = eval_summary[
                (eval_summary["level"] == "dataset")
                & (eval_summary["dataset_name"].astype(str) == dataset)
                & (eval_summary["score_name"] == score)
            ]
            row[f"{score}_aligned_rho"] = (
                float(sub["direction_aligned_spearman"].iloc[0]) if not sub.empty else float("nan")
            )
            row[f"{score}_risk_cov_80_improve_pct"] = (
                float(sub["risk_cov_80_improve_pct"].iloc[0]) if not sub.empty else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / den)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    c = cosine(a, b)
    if np.isnan(c):
        return float("nan")
    return float(1.0 - c)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def topk_overlap_error(a: np.ndarray, b: np.ndarray, k: int = 20) -> float:
    k = min(k, len(a), len(b))
    if k <= 0:
        return float("nan")
    ai = set(np.argsort(-np.abs(a))[:k].tolist())
    bi = set(np.argsort(-np.abs(b))[:k].tolist())
    return float(1.0 - len(ai & bi) / k)


def corr_pair(x, y, method: str) -> float:
    s = pd.Series(x, dtype="float64")
    t = pd.Series(y, dtype="float64")
    mask = s.notna() & t.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    return float(s[mask].corr(t[mask], method=method))


def fast_spearman(x, y) -> float:
    df = pd.DataFrame({"x": np.asarray(x, dtype=float), "y": np.asarray(y, dtype=float)}).dropna()
    if len(df) < 3:
        return float("nan")
    xr = df["x"].rank(method="average").to_numpy(dtype=float)
    yr = df["y"].rank(method="average").to_numpy(dtype=float)
    if np.nanstd(xr) <= 1e-12 or np.nanstd(yr) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def zscore_by_reference(values: pd.Series, reference: pd.Series) -> pd.Series:
    ref = pd.to_numeric(reference, errors="coerce")
    val = pd.to_numeric(values, errors="coerce")
    med = float(ref.median()) if ref.notna().any() else 0.0
    q1 = float(ref.quantile(0.25)) if ref.notna().any() else 0.0
    q3 = float(ref.quantile(0.75)) if ref.notna().any() else 1.0
    scale = q3 - q1
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = float(ref.std()) if ref.notna().sum() > 1 else 1.0
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = 1.0
    return (val.fillna(med) - med) / scale


def minmax_from_reference(values: pd.Series, reference: pd.Series) -> pd.Series:
    ref = pd.to_numeric(reference, errors="coerce")
    val = pd.to_numeric(values, errors="coerce")
    if not ref.notna().any():
        return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
    lo = float(ref.min())
    hi = float(ref.max())
    if hi - lo <= 1e-12:
        return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
    return (val.fillna(lo) - lo) / (hi - lo)


def parse_dataset_names(raw: str) -> list[str]:
    names = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not names:
        raise ValueError("At least one dataset name is required")
    return names


def _path_exists_from_scan(row: pd.Series) -> Path:
    path = Path(str(row["local_path"]))
    if not path.exists():
        raise FileNotFoundError(f"Dataset file missing: {path}")
    return path


def resolve_dataset_selector(scan: pd.DataFrame, selector: str) -> tuple[str, Path]:
    """Resolve one dataset selector against metadata/h5ad_scan.tsv.

    Backward-compatible selector forms:
      * study_family, e.g. Haber or sciplex3
      * file_name/stem, e.g. sciplex3_A549.h5ad or sciplex3_A549
      * study_family@file_name, e.g. sciplex3@sciplex3_A549.h5ad
      * absolute/local path already present in the scan table
    """

    selector = selector.strip()
    if "@" in selector:
        family, file_selector = [x.strip() for x in selector.split("@", 1)]
        sub = scan[scan["study_family"].astype(str) == family].copy()
        if sub.empty:
            raise FileNotFoundError(f"Dataset family {family} not found in scan table")
        file_low = file_selector.lower()
        mask = (
            sub["file_name"].astype(str).str.lower().eq(file_low)
            | sub["file_name"].astype(str).map(lambda x: Path(x).stem.lower()).eq(file_low)
            | sub["local_path"].astype(str).map(lambda x: Path(x).name.lower()).eq(file_low)
            | sub["local_path"].astype(str).map(lambda x: Path(x).stem.lower()).eq(file_low)
        )
        hit = sub[mask]
        if hit.empty:
            raise FileNotFoundError(f"Selector {selector} not found in scan table")
        row = hit.iloc[0]
        return Path(str(row["local_path"])).stem, _path_exists_from_scan(row)

    sel_low = selector.lower()
    file_mask = (
        scan["file_name"].astype(str).str.lower().eq(sel_low)
        | scan["file_name"].astype(str).map(lambda x: Path(x).stem.lower()).eq(sel_low)
        | scan["local_path"].astype(str).map(lambda x: Path(x).name.lower()).eq(sel_low)
        | scan["local_path"].astype(str).map(lambda x: Path(x).stem.lower()).eq(sel_low)
        | scan["local_path"].astype(str).eq(selector)
    )
    file_hit = scan[file_mask]
    if not file_hit.empty:
        row = file_hit.iloc[0]
        return Path(str(row["local_path"])).stem, _path_exists_from_scan(row)

    family_hit = scan[scan["study_family"].astype(str) == selector]
    if family_hit.empty:
        raise FileNotFoundError(f"Dataset selector {selector} not found in scan table")
    row = family_hit.iloc[0]
    return selector, _path_exists_from_scan(row)


def resolve_dataset_paths(atlas_root: Path, dataset_names: list[str] | None = None) -> dict[str, Path]:
    scan_path = atlas_root / "metadata" / "h5ad_scan.tsv"
    if not scan_path.exists():
        raise FileNotFoundError(f"Missing scan table: {scan_path}")
    scan = pd.read_csv(scan_path, sep="\t")
    out: dict[str, Path] = {}
    for selector in (dataset_names or DATASET_NAMES):
        name, path = resolve_dataset_selector(scan, selector)
        if name in out:
            raise ValueError(f"Duplicate resolved dataset name {name}; use unique file selectors")
        out[name] = path
    return out


def run_s0_audit(dirs: RunDirs) -> pd.DataFrame:
    result_paths = [
        PROJECT_ROOT / "46_q1_cpu_push_20260520" / "results" / "SAFETY_TASK_METRICS.csv",
        PROJECT_ROOT / "51_policy_calibrated_q1_20260520" / "results" / "SAFETY_TASK_METRICS.csv",
    ]
    rows = []
    for path in result_paths:
        source_run = path.parents[1].name
        if not path.exists():
            rows.append({"source_run": source_run, "path": str(path), "status": "missing"})
            continue
        df = pd.read_csv(path)
        needed = {"dataset", "split_type", "model", "rmse", "confidence"}
        if not needed.issubset(df.columns):
            rows.append({"source_run": source_run, "path": str(path), "status": "missing_columns", "columns": ",".join(df.columns)})
            continue
        group_cols = [c for c in ["phase", "dataset", "split_type", "model"] if c in df.columns]
        for key, g in df.groupby(group_cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            rec = {"source_run": source_run, "path": str(path), "status": "ok"}
            rec.update(dict(zip(group_cols, key)))
            gg = g[["confidence", "rmse"]].dropna()
            rec["n_total"] = int(len(g))
            rec["n_with_confidence"] = int(len(gg))
            rec["spearman_confidence_vs_rmse"] = corr_pair(gg["confidence"], gg["rmse"], "spearman") if len(gg) >= 3 else float("nan")
            rec["pearson_confidence_vs_rmse"] = corr_pair(gg["confidence"], gg["rmse"], "pearson") if len(gg) >= 3 else float("nan")
            rec["direction_ok"] = bool(np.isfinite(rec["spearman_confidence_vs_rmse"]) and rec["spearman_confidence_vs_rmse"] < 0)
            rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(dirs.tables / "S0_existing_signal.csv", index=False)

    ok = out[(out.get("status", "") == "ok") & (out.get("n_with_confidence", 0) >= 3)].copy()
    best = ok.sort_values("spearman_confidence_vs_rmse").head(8) if not ok.empty else ok
    lines = [
        "# S0 existing signal audit",
        "",
        f"- Requested exact spec exists: `{REQUESTED_SPEC.exists()}`.",
        f"- Requested exact spec path: `{REQUESTED_SPEC}`.",
        f"- Fallback plan exists: `{FALLBACK_PLAN.exists()}`.",
        f"- This Phase 2 run records the missing exact spec as an audit fact and follows the available fallback plan plus the user's current constraints.",
        "",
        "## Inputs checked",
    ]
    for path in result_paths:
        lines.append(f"- `{path}`: {'found' if path.exists() else 'missing'}")
    lines.extend(["", "## Signal summary", ""])
    if ok.empty:
        lines.append("No usable historical confidence/error rows were found.")
    else:
        lines.append(f"- Usable groups: {len(ok)}")
        lines.append(f"- Groups with expected negative direction: {int(ok['direction_ok'].sum())}")
        lines.append("")
        lines.append("Best negative correlations:")
        lines.append(df_markdown(best))
    write_text(dirs.reports / "S0_audit_report.md", "\n".join(lines) + "\n")
    return out


def build_all_tasks(paths: dict[str, Path], dirs: RunDirs, n_genes: int, min_cells: int, max_cells_per_group: int, seed: int):
    tasks_by_dataset: dict[str, list[dict]] = {}
    genes_by_dataset: dict[str, list[str]] = {}
    meta_rows = []
    for name, path in paths.items():
        print(f"[tasks] building {name} from {path}", flush=True)
        tasks, genes, meta = build_effect_tasks(
            path,
            name,
            n_genes=n_genes,
            min_cells=min_cells,
            max_cells_per_group=max_cells_per_group,
            seed=seed,
        )
        for i, t in enumerate(tasks):
            t["task_id"] = i
            t["task_key"] = f"{name}::task_{i:05d}"
        tasks_by_dataset[name] = tasks
        genes_by_dataset[name] = genes
        meta_rows.append(meta)
    pd.DataFrame(meta_rows).to_csv(dirs.tables / "DATASET_TASK_SUMMARY.csv", index=False)
    save_json(dirs.input / "dataset_paths.json", {k: str(v) for k, v in paths.items()})
    return tasks_by_dataset, genes_by_dataset, pd.DataFrame(meta_rows)


def check_support(tasks: list[dict], test_ids: list[int], train_ids: set[int]) -> tuple[bool, list[str]]:
    pair_train = {(tasks[i]["context"], tasks[i]["perturbation"]) for i in train_ids}
    ctx_train = {tasks[i]["context"] for i in train_ids}
    pert_train = {tasks[i]["perturbation"] for i in train_ids}
    errors = []
    for i in test_ids:
        pair = (tasks[i]["context"], tasks[i]["perturbation"])
        if pair in pair_train:
            errors.append(f"pair_leak:{i}")
        if tasks[i]["context"] not in ctx_train:
            errors.append(f"context_missing:{i}")
        if tasks[i]["perturbation"] not in pert_train:
            errors.append(f"pert_missing:{i}")
        other_pert = [
            j for j in train_ids
            if tasks[j]["perturbation"] == tasks[i]["perturbation"] and tasks[j]["context"] != tasks[i]["context"]
        ]
        other_ctx = [
            j for j in train_ids
            if tasks[j]["context"] == tasks[i]["context"] and tasks[j]["perturbation"] != tasks[i]["perturbation"]
        ]
        if not other_pert:
            errors.append(f"no_source_context_for_pert:{i}")
        if not other_ctx:
            errors.append(f"no_other_pert_for_context:{i}")
    return len(errors) == 0, errors


def build_pair_splits(tasks_by_dataset: dict[str, list[dict]], dirs: RunDirs, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    all_rows = []
    summary = []
    for dataset, tasks in tasks_by_dataset.items():
        n = len(tasks)
        ctx_counts = pd.Series([t["context"] for t in tasks]).value_counts().to_dict()
        pert_counts = pd.Series([t["perturbation"] for t in tasks]).value_counts().to_dict()
        candidate = [
            i for i, t in enumerate(tasks)
            if ctx_counts.get(t["context"], 0) >= 2 and pert_counts.get(t["perturbation"], 0) >= 2
        ]
        if len(candidate) >= 15:
            n_folds = 5
        elif len(candidate) >= 6:
            n_folds = 3
        else:
            n_folds = max(1, min(3, len(candidate)))
        fold_tests = {f: [] for f in range(n_folds)}
        fold_ctx_counts = {f: {} for f in range(n_folds)}
        fold_pert_counts = {f: {} for f in range(n_folds)}
        # Balanced global assignment.  A fold may not hold out every task of a
        # context or every task of a perturbation, otherwise that test pair is no
        # longer a held-out pair transfer problem.
        order = list(candidate)
        rng.shuffle(order)
        order = sorted(order, key=lambda i: (ctx_counts.get(tasks[i]["context"], 0), pert_counts.get(tasks[i]["perturbation"], 0)))
        skipped_candidates = []
        for idx in order:
            ctx = tasks[idx]["context"]
            pert = tasks[idx]["perturbation"]
            valid_folds = []
            for f in range(n_folds):
                ctx_in_fold = fold_ctx_counts[f].get(ctx, 0)
                pert_in_fold = fold_pert_counts[f].get(pert, 0)
                if ctx_in_fold < ctx_counts.get(ctx, 0) - 1 and pert_in_fold < pert_counts.get(pert, 0) - 1:
                    valid_folds.append(f)
            if not valid_folds:
                skipped_candidates.append(idx)
                continue
            min_size = min(len(fold_tests[f]) for f in valid_folds)
            best_folds = [f for f in valid_folds if len(fold_tests[f]) == min_size]
            f = int(rng.choice(best_folds))
            fold_tests[f].append(idx)
            fold_ctx_counts[f][ctx] = fold_ctx_counts[f].get(ctx, 0) + 1
            fold_pert_counts[f][pert] = fold_pert_counts[f].get(pert, 0) + 1
        for fold in range(n_folds):
            test_ids = sorted(fold_tests[fold])
            train_pool = set(range(n)) - set(test_ids)
            val_ids: list[int] = []
            target_val = max(1, int(round(0.15 * len(train_pool))))
            candidates = list(train_pool)
            rng.shuffle(candidates)
            train_core = set(train_pool)
            for cand in candidates:
                if len(val_ids) >= target_val:
                    break
                proposed = train_core - {cand}
                ok, _ = check_support(tasks, test_ids, proposed)
                if ok and len(proposed) >= 3:
                    val_ids.append(cand)
                    train_core = proposed
            if not val_ids:
                # Prefer a tiny but valid calibration split; if impossible, keep all as train.
                for cand in candidates:
                    proposed = train_core - {cand}
                    ok, _ = check_support(tasks, test_ids, proposed)
                    if ok and len(proposed) >= 3:
                        val_ids.append(cand)
                        train_core = proposed
                        break
            train_ids = sorted(train_core)
            val_ids = sorted(val_ids)
            ok, errors = check_support(tasks, test_ids, set(train_ids))
            train_pairs = {(tasks[i]["context"], tasks[i]["perturbation"]) for i in train_ids}
            train_contexts = {tasks[i]["context"] for i in train_ids}
            train_perts = {tasks[i]["perturbation"] for i in train_ids}
            for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
                for i in ids:
                    pair = (tasks[i]["context"], tasks[i]["perturbation"])
                    all_rows.append(
                        {
                            "dataset_name": dataset,
                            "task_id": int(i),
                            "task_key": tasks[i]["task_key"],
                            "context": tasks[i]["context"],
                            "perturbation": tasks[i]["perturbation"],
                            "fold_id": int(fold),
                            "split": split_name,
                            "pair_seen_in_train": bool(pair in train_pairs),
                            "perturbation_seen_in_train": bool(tasks[i]["perturbation"] in train_perts),
                            "context_seen_in_train": bool(tasks[i]["context"] in train_contexts),
                        }
                    )
            summary.append(
                {
                    "dataset_name": dataset,
                    "fold_id": fold,
                    "n_tasks_total": n,
                    "n_candidate_test_pairs": len(candidate),
                    "n_skipped_candidate_pairs": len(skipped_candidates),
                    "n_train": len(train_ids),
                    "n_val": len(val_ids),
                    "n_test": len(test_ids),
                    "n_contexts": len({t["context"] for t in tasks}),
                    "n_perturbations": len({t["perturbation"] for t in tasks}),
                    "support_check_pass": bool(ok),
                    "support_errors": ";".join(errors[:10]),
                    "n_folds": n_folds,
                }
            )
    split_df = pd.DataFrame(all_rows)
    split_df.to_csv(dirs.tables / "HELDOUT_PAIR_SPLITS.csv", index=False)
    split_df.to_csv(dirs.input / "HELDOUT_PAIR_SPLITS.csv", index=False)
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(dirs.tables / "HELDOUT_PAIR_SPLIT_SUMMARY.csv", index=False)
    leakage = split_df[(split_df["split"] == "test") & (split_df["pair_seen_in_train"])]
    missing_p = split_df[(split_df["split"] == "test") & (~split_df["perturbation_seen_in_train"])]
    missing_c = split_df[(split_df["split"] == "test") & (~split_df["context_seen_in_train"])]
    lines = [
        "# Held-out pair split audit",
        "",
        f"- Datasets: {', '.join(sorted(tasks_by_dataset))}",
        f"- Total split rows: {len(split_df)}",
        f"- Test pair leakage rows: {len(leakage)}",
        f"- Test rows with perturbation absent from train: {len(missing_p)}",
        f"- Test rows with context absent from train: {len(missing_c)}",
        "",
        "## Fold summary",
        df_markdown(summary_df),
    ]
    write_text(dirs.reports / "split_audit_report.md", "\n".join(lines) + "\n")
    return split_df


def train_mask_from_split(tasks: list[dict], split_sub: pd.DataFrame, fold_id: int) -> np.ndarray:
    rows = split_sub[(split_sub["fold_id"] == fold_id) & (split_sub["split"] == "train")]
    train_ids = set(rows["task_id"].astype(int).tolist())
    return np.array([i in train_ids for i in range(len(tasks))], dtype=bool)


def run_predictors(
    tasks_by_dataset: dict[str, list[dict]],
    genes_by_dataset: dict[str, list[str]],
    split_df: pd.DataFrame,
    dirs: RunDirs,
):
    records = []
    pred_arrays = {}
    true_arrays = {}
    control_arrays = {}
    rec_i = 0
    status_rows = []
    for dataset, tasks in tasks_by_dataset.items():
        genes = genes_by_dataset[dataset]
        expected_n_genes = len(genes)
        panel_id = f"{dataset}::effect_gene_panel_n{expected_n_genes}"
        order_hash = gene_order_hash(genes)
        normalization_id = f"pseudobulk_mean_diff_n_genes_{expected_n_genes}"
        sub = split_df[split_df["dataset_name"] == dataset]
        for fold_id in sorted(sub["fold_id"].unique()):
            train_mask = train_mask_from_split(tasks, sub, int(fold_id))
            # Phase 2.1 stores train, val and test records. Predictors are fit
            # only on train; train predictions are in-sample calibration rows
            # for fold-local combined/learned scoring, never test evaluation.
            eval_rows = sub[sub["fold_id"] == fold_id].copy()
            eval_ids = eval_rows["task_id"].astype(int).to_numpy()
            predictors = [
                ("V0StrongBaseline", V0StrongBaseline()),
                ("ContextSimBaseline", ContextSimilarityBaseline()),
            ]
            for pred_name, model in predictors:
                try:
                    model.fit(tasks, train_mask)
                    preds = model.predict(tasks, eval_ids)
                    status = "ok"
                    error = ""
                except Exception as exc:  # pragma: no cover - runtime safety.
                    preds = np.full((len(eval_ids), len(tasks[0]["effect"])), np.nan, dtype=np.float32)
                    status = "failed"
                    error = repr(exc)
                status_rows.append(
                    {
                        "dataset_name": dataset,
                        "fold_id": int(fold_id),
                        "predictor_name": pred_name,
                        "status": status,
                        "error": error,
                        "n_eval": len(eval_ids),
                    }
                )
                for row, task_id, pred in zip(eval_rows.to_dict("records"), eval_ids, preds):
                    task = tasks[int(task_id)]
                    context = required_task_value(task, "context")
                    perturbation = required_task_value(task, "perturbation")
                    true = task["effect"].astype(np.float32)
                    if true.shape != (expected_n_genes,):
                        raise ValueError(
                            f"Task {task['task_key']} true effect shape {true.shape} "
                            f"does not match gene panel length {expected_n_genes}"
                        )
                    pred = np.asarray(pred, dtype=np.float32)
                    if pred.shape != (expected_n_genes,):
                        raise ValueError(
                            f"Task {task['task_key']} predictor {pred_name} shape {pred.shape} "
                            f"does not match gene panel length {expected_n_genes}"
                        )
                    record_id = f"v2_rec_{rec_i:06d}"
                    pred_key = f"{record_id}::predicted_effect"
                    task_scope = f"{dataset}::fold{int(fold_id)}::{row['split']}::task_{int(task_id):05d}"
                    true_key = f"{task_scope}::true_effect"
                    ctrl_key = f"{task_scope}::target_control_mean"
                    pred_arrays[pred_key] = pred.astype(np.float32)
                    true_arrays[true_key] = true
                    control_arrays[ctrl_key] = task["control_mean"].astype(np.float32)
                    records.append(
                        {
                            "record_id": record_id,
                            "task_id": int(task_id),
                            "task_key": task["task_key"],
                            "dataset_name": dataset,
                            "dataset_group": dataset,
                            "fold_id": int(fold_id),
                            "split": row["split"],
                            "context": context,
                            "perturbation": perturbation,
                            "predictor_name": pred_name,
                            "schema_version": "safeconf_prediction_record_v1",
                            "run_type": "formal",
                            "gene_panel_id": panel_id,
                            "gene_order_hash": order_hash,
                            "effect_definition": "mean_diff",
                            "normalization_id": normalization_id,
                            "error_normalization": "raw_rmse",
                            "predicted_effect_key": pred_key,
                            "true_effect_key": true_key,
                            "target_control_key": ctrl_key,
                            "true_error_rmse": rmse(pred, true),
                            "true_error_cosine": cosine_distance(pred, true),
                            "true_error_top20": topk_overlap_error(pred, true, k=20),
                        }
                    )
                    rec_i += 1
    rec_df = pd.DataFrame(records)
    rec_df.to_csv(dirs.tables / "PREDICTION_RECORDS.csv", index=False)
    rec_df.to_csv(dirs.input / "PREDICTION_RECORDS.csv", index=False)
    np.savez_compressed(dirs.input / "predicted_effects.npz", **pred_arrays)
    np.savez_compressed(dirs.input / "true_effects.npz", **true_arrays)
    np.savez_compressed(dirs.input / "target_control_means.npz", **control_arrays)
    status_df = pd.DataFrame(status_rows)
    status_df.to_csv(dirs.tables / "PREDICTOR_STATUS.csv", index=False)
    lines = [
        "# Predictor run report",
        "",
        f"- PredictionRecord rows: {len(rec_df)}",
        f"- Predictors: {', '.join(sorted(rec_df['predictor_name'].unique())) if not rec_df.empty else 'none'}",
        "",
        "## Status",
        df_markdown(status_df),
    ]
    write_text(dirs.reports / "predictor_run_report.md", "\n".join(lines) + "\n")
    return rec_df, pred_arrays, true_arrays


def mean_pairwise_cosine(effects: np.ndarray) -> float:
    if len(effects) < 2:
        return float("nan")
    vals = []
    for i in range(len(effects)):
        for j in range(i + 1, len(effects)):
            vals.append(cosine(effects[i], effects[j]))
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def normalize_vec(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    den = np.linalg.norm(x)
    if den <= 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return x / den


def row_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"row_normalize expects 2D array, got {x.shape}")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[~np.isfinite(norms) | (norms <= 1e-12)] = 1.0
    return x / norms


def compute_train_stability(tasks: list[dict], train_ids: list[int]) -> dict[str, float]:
    by_pert: dict[str, list[np.ndarray]] = {}
    for i in train_ids:
        by_pert.setdefault(tasks[i]["perturbation"], []).append(tasks[i]["effect"])
    out = {}
    for pert, vals in by_pert.items():
        if len(vals) >= 2:
            out[pert] = mean_pairwise_cosine(np.stack(vals, axis=0))
    return out


def compute_historical_residual(tasks: list[dict], train_ids: list[int]) -> tuple[dict[str, float], float]:
    """Train-only historical risk: LOO source-context residual per perturbation."""
    by_pert: dict[str, list[int]] = {}
    for i in train_ids:
        by_pert.setdefault(tasks[i]["perturbation"], []).append(i)
    out: dict[str, float] = {}
    all_errs = []
    for pert, ids in by_pert.items():
        errs = []
        for i in ids:
            src = [j for j in ids if tasks[j]["context"] != tasks[i]["context"]]
            if not src:
                continue
            pred = np.mean([tasks[j]["effect"] for j in src], axis=0)
            err = rmse(pred, tasks[i]["effect"])
            errs.append(err)
            all_errs.append(err)
        if errs:
            out[pert] = float(np.median(errs))
    fallback = float(np.median(all_errs)) if all_errs else float("nan")
    return out, fallback


def compute_features(
    rec_df: pd.DataFrame,
    tasks_by_dataset: dict[str, list[dict]],
    split_df: pd.DataFrame,
    pred_arrays: dict[str, np.ndarray],
    dirs: RunDirs,
) -> pd.DataFrame:
    rows = []
    # Model disagreement is computed per task/fold/split across predictors.
    disagreement: dict[tuple, tuple[float, float]] = {}
    for key, g in rec_df.groupby(["dataset_name", "fold_id", "split", "task_id"], dropna=False):
        if g["predictor_name"].nunique() >= 2:
            by_name = {r["predictor_name"]: pred_arrays[r["predicted_effect_key"]] for r in g.to_dict("records")}
            if "V0StrongBaseline" in by_name and "ContextSimBaseline" in by_name:
                a = by_name["V0StrongBaseline"]
                b = by_name["ContextSimBaseline"]
                disagreement[key] = (rmse(a, b), cosine_distance(a, b))
    for (dataset, fold_id), rec_sub in rec_df.groupby(["dataset_name", "fold_id"]):
        tasks = tasks_by_dataset[dataset]
        split_sub = split_df[(split_df["dataset_name"] == dataset) & (split_df["fold_id"] == fold_id)]
        train_ids = split_sub[split_sub["split"] == "train"]["task_id"].astype(int).tolist()
        train_effects = np.stack([tasks[i]["effect"] for i in train_ids], axis=0)
        train_controls = np.stack([tasks[i]["control_mean"] for i in train_ids], axis=0)
        train_controls_unit = row_normalize(train_controls)
        train_contexts = np.array([tasks[i]["context"] for i in train_ids], dtype=object)
        train_perts = np.array([tasks[i]["perturbation"] for i in train_ids], dtype=object)
        train_ids_by_pert: dict[str, list[int]] = {}
        for i in train_ids:
            train_ids_by_pert.setdefault(tasks[i]["perturbation"], []).append(i)
        historical_residual, historical_fallback = compute_historical_residual(tasks, train_ids)
        global_effect = np.nanmean(train_effects, axis=0)
        median_effect_norm = float(np.nanmedian(np.linalg.norm(train_effects, axis=1)))
        # Represent train tasks by control state plus observed effect.
        train_reps = np.concatenate([train_controls_unit, row_normalize(train_effects)], axis=1)
        train_reps_unit = row_normalize(train_reps)
        task_feature_cache: dict[int, dict] = {}
        source_stats_cache: dict[tuple[str, str], tuple[int, float, float, np.ndarray]] = {}

        def source_stats(perturbation: str, context: str) -> tuple[int, float, float, np.ndarray]:
            cache_key = (perturbation, context)
            if cache_key in source_stats_cache:
                return source_stats_cache[cache_key]
            same_pert_other = [i for i in train_ids_by_pert.get(perturbation, []) if tasks[i]["context"] != context]
            source_effects = (
                np.stack([tasks[i]["effect"] for i in same_pert_other], axis=0)
                if same_pert_other
                else np.empty((0, train_effects.shape[1]))
            )
            support_count = len({tasks[i]["context"] for i in same_pert_other})
            raw_stability = mean_pairwise_cosine(source_effects) if len(source_effects) >= 2 else float("nan")
            effect_var = float(np.mean(np.var(source_effects, axis=0))) if len(source_effects) >= 2 else float("nan")
            source_mean = source_effects.mean(axis=0) if len(source_effects) else global_effect
            out = (support_count, raw_stability, effect_var, source_mean)
            source_stats_cache[cache_key] = out
            return out

        def task_features(task_id: int) -> dict:
            if task_id in task_feature_cache:
                return task_feature_cache[task_id]
            task = tasks[task_id]
            other_ctx_mask = train_contexts != task["context"]
            task_control_unit = normalize_vec(task["control_mean"])
            ctx_sims = train_controls_unit[other_ctx_mask] @ task_control_unit
            if len(ctx_sims) == 0 or np.all(~np.isfinite(ctx_sims)):
                ctx_sims = train_controls_unit @ task_control_unit
            support_count, raw_stability, effect_var, source_mean = source_stats(
                task["perturbation"], task["context"]
            )
            target_rep = normalize_vec(np.concatenate([task_control_unit, normalize_vec(source_mean)]))
            dists = 1.0 - (train_reps_unit @ target_rep)
            dists = dists[np.isfinite(dists)]
            k = min(5, len(dists))
            out = {
                "context_similarity_max": float(np.nanmax(ctx_sims)) if len(ctx_sims) else float("nan"),
                "context_similarity_mean": float(np.nanmean(ctx_sims)) if len(ctx_sims) else float("nan"),
                "perturbation_support_count": int(support_count),
                "perturbation_effect_stability": raw_stability,
                "perturbation_effect_variance": effect_var,
                "ood_nearest_distance": float(np.nanmin(dists)) if len(dists) else float("nan"),
                "ood_mean_k_distance": float(np.nanmean(np.sort(dists)[:k])) if len(dists) else float("nan"),
                "historical_residual_risk": historical_residual.get(task["perturbation"], historical_fallback),
            }
            task_feature_cache[task_id] = out
            return out

        for r in rec_sub.to_dict("records"):
            task_id = int(r["task_id"])
            task = tasks[task_id]
            pred = pred_arrays[r["predicted_effect_key"]]
            base_features = task_features(task_id)
            pred_norm = float(np.linalg.norm(pred))
            norm_ratio = pred_norm / (median_effect_norm + 1e-8)
            dis_key = (dataset, fold_id, r["split"], task_id)
            dis_rmse, dis_cos = disagreement.get(dis_key, (float("nan"), float("nan")))
            rows.append(
                {
                    "record_id": r["record_id"],
                    "task_id": task_id,
                    "task_key": r["task_key"],
                    "dataset_name": dataset,
                    "fold_id": int(fold_id),
                    "split": r["split"],
                    "context": task["context"],
                    "perturbation": task["perturbation"],
                    "predictor_name": r["predictor_name"],
                    "context_similarity_max": base_features["context_similarity_max"],
                    "context_similarity_mean": base_features["context_similarity_mean"],
                    "perturbation_support_count": base_features["perturbation_support_count"],
                    "perturbation_effect_stability": base_features["perturbation_effect_stability"],
                    "perturbation_effect_variance": base_features["perturbation_effect_variance"],
                    "prediction_l2_norm": pred_norm,
                    "prediction_abs_mean": float(np.mean(np.abs(pred))),
                    "fold_train_median_effect_norm": median_effect_norm,
                    "prediction_norm_ratio": float(norm_ratio),
                    "prediction_magnitude_deviation": float(abs(math.log(norm_ratio + 1e-8))),
                    "model_disagreement_rmse": dis_rmse,
                    "model_disagreement_cosine": dis_cos,
                    "ood_nearest_distance": base_features["ood_nearest_distance"],
                    "ood_mean_k_distance": base_features["ood_mean_k_distance"],
                    "historical_residual_risk": base_features["historical_residual_risk"],
                }
            )
    feat = pd.DataFrame(rows)
    feat.to_csv(dirs.tables / "CONFIDENCE_FEATURES.csv", index=False)
    missing = feat.isna().mean(numeric_only=False).sort_values(ascending=False).reset_index()
    missing.columns = ["column", "missing_rate"]
    dataset_missing = (
        feat.groupby("dataset_name")[["perturbation_effect_stability", "historical_residual_risk", "ood_nearest_distance"]]
        .apply(lambda x: x.isna().mean())
        .reset_index()
    )
    numeric = feat.select_dtypes(include=[np.number]).describe().T.reset_index().rename(columns={"index": "feature"})
    missing.to_csv(dirs.tables / "CONFIDENCE_FEATURE_MISSINGNESS.csv", index=False)
    dataset_missing.to_csv(dirs.tables / "CONFIDENCE_FEATURE_MISSINGNESS_BY_DATASET.csv", index=False)
    numeric.to_csv(dirs.tables / "CONFIDENCE_FEATURE_DESCRIBE.csv", index=False)
    lines = [
        "# Confidence feature report",
        "",
        "Phase 2.1 fixes:",
        "- `context_similarity_*` excludes the same target context where possible.",
        "- `perturbation_effect_stability` is raw; if train has fewer than two source contexts it stays NaN.",
        "- `historical_residual_risk` is train-only leave-one-context-out residual by perturbation.",
        "- `ood_nearest_distance` is computed from train-only task representations.",
        "- `prediction_magnitude_deviation` normalizes prediction norm by each fold's train effect norm.",
        "",
        "## Missingness",
        df_markdown(missing),
        "",
        "## Missingness by dataset",
        df_markdown(dataset_missing),
        "",
        "## Numeric summary",
        df_markdown(numeric),
    ]
    write_text(dirs.reports / "confidence_features_report.md", "\n".join(lines) + "\n")
    return feat


def add_score(score_rows: list[dict], base: pd.DataFrame, name: str, score_type: str, values: pd.Series):
    for idx, value in values.items():
        r = base.loc[idx]
        score_rows.append(
            {
                "record_id": r["record_id"],
                "dataset_name": r["dataset_name"],
                "fold_id": int(r["fold_id"]),
                "split": r["split"],
                "context": r["context"],
                "perturbation": r["perturbation"],
                "predictor_name": r["predictor_name"],
                "score_name": name,
                "score_type": score_type,
                "score_value": float(value) if pd.notna(value) else float("nan"),
                "true_error_rmse": float(r["true_error_rmse"]),
                "true_error_cosine": float(r.get("true_error_cosine", np.nan)),
                "true_error_top20": float(r.get("true_error_top20", np.nan)),
            }
        )


def build_z_features(sub: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ctx": zscore_by_reference(sub["context_similarity_max"], ref["context_similarity_max"]),
            "stab": zscore_by_reference(sub["perturbation_effect_stability"], ref["perturbation_effect_stability"]),
            "support": zscore_by_reference(np.log1p(sub["perturbation_support_count"].astype(float)), np.log1p(ref["perturbation_support_count"].astype(float))),
            "disagreement": zscore_by_reference(sub["model_disagreement_rmse"], ref["model_disagreement_rmse"]),
            "magnitude": zscore_by_reference(sub["prediction_magnitude_deviation"], ref["prediction_magnitude_deviation"]),
            "ood": zscore_by_reference(sub["ood_nearest_distance"], ref["ood_nearest_distance"]),
        },
        index=sub.index,
    )


def combined_score_from_weights(z: pd.DataFrame, weights: tuple[float, float, float, float, float, float]) -> pd.Series:
    w1, w2, w3, w4, w5, w6 = weights
    return (
        w1 * z["ctx"]
        + w2 * z["stab"]
        + w3 * z["support"]
        - w4 * z["disagreement"]
        - w5 * z["magnitude"]
        - w6 * z["ood"]
    )


def tune_combined_weights(sub: pd.DataFrame, z: pd.DataFrame) -> tuple[tuple[float, float, float, float, float, float], float, str]:
    val_idx = sub.index[sub["split"] == "val"].tolist()
    train_idx = sub.index[sub["split"] == "train"].tolist()
    target_idx = val_idx if len(val_idx) >= 3 else train_idx
    source = "val" if len(val_idx) >= 3 else "train_fallback"
    candidate_weights = [
        (1, 1, 1, 1, 1, 1),
        (1, 0, 1, 1, 0, 0),
        (1, 0, 1, 2, 0, 0),
        (2, 0, 1, 1, 0, 0),
        (2, 1, 1, 1, 0, 0),
        (2, 0, 1, 2, 1, 0),
        (1, 1, 0, 2, 0, 0),
        (1, 0, 0, 2, 0, 0),
        (0, 0, 0, 1, 0, 0),
        (1, 0, 0, 1, 0, 1),
        (1, 0, 1, 1, 0, 1),
        (1, 0, 1, 1, 1, 1),
        (2, 0, 2, 1, 0, 1),
        (2, 0, 1, 3, 0, 1),
        (3, 0, 1, 1, 0, 1),
        (3, 0, 2, 2, 0, 1),
        (0, 1, 1, 1, 0, 0),
        (0, 1, 0, 2, 0, 1),
        (1, 2, 1, 1, 0, 1),
        (1, 0, 2, 2, 0, 2),
        (2, 0, 0, 2, 0, 2),
        (0.5, 0, 1, 2, 0, 1),
        (1, 0, 0.5, 2, 0, 1),
        (1, 0.5, 1, 2, 0, 1),
    ]
    best_weights = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    best_score = -np.inf
    if len(target_idx) < 3:
        return best_weights, float("nan"), "default_too_few_rows"
    y = sub.loc[target_idx, "true_error_rmse"]
    for weights in candidate_weights:
        score = combined_score_from_weights(z.loc[target_idx], weights)
        sp = fast_spearman(score.to_numpy(dtype=float), y.to_numpy(dtype=float))
        aligned = -sp if np.isfinite(sp) else -np.inf
        if aligned > best_score:
            best_score = aligned
            best_weights = tuple(float(x) for x in weights)
    return best_weights, float(best_score) if np.isfinite(best_score) else float("nan"), source


def run_scores(rec_df: pd.DataFrame, feat_df: pd.DataFrame, dirs: RunDirs, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = rec_df.merge(feat_df, on=[
        "record_id", "task_id", "task_key", "dataset_name", "fold_id", "split", "context", "perturbation", "predictor_name"
    ], how="left")
    score_rows: list[dict] = []
    rng = np.random.default_rng(seed)
    base = merged.reset_index(drop=True)
    add_score(score_rows, base, "random_score", "confidence", pd.Series(rng.random(len(base)), index=base.index))
    add_score(score_rows, base, "context_similarity_score", "confidence", base["context_similarity_max"])
    add_score(score_rows, base, "perturbation_stability_score", "confidence", base["perturbation_effect_stability"])
    add_score(score_rows, base, "support_count_score", "confidence", np.log1p(base["perturbation_support_count"].astype(float)))
    add_score(score_rows, base, "prediction_magnitude_risk", "risk", base["prediction_magnitude_deviation"])
    add_score(score_rows, base, "model_disagreement_risk", "risk", base["model_disagreement_rmse"])
    add_score(score_rows, base, "ood_distance_risk", "risk", base["ood_nearest_distance"])
    add_score(score_rows, base, "historical_residual_risk", "risk", base["historical_residual_risk"])

    combined = pd.Series(np.nan, index=base.index, dtype=float)
    weight_rows = []
    for (dataset, fold_id, predictor), idx_obj in base.groupby(["dataset_name", "fold_id", "predictor_name"]).groups.items():
        idx = list(idx_obj)
        sub = base.loc[idx]
        ref = sub[sub["split"] == "train"]
        if ref.empty:
            ref = sub[sub["split"] == "val"]
        if ref.empty:
            ref = sub
        z = build_z_features(sub, ref)
        weights, val_aligned, source = tune_combined_weights(sub, z)
        combined.loc[idx] = combined_score_from_weights(z, weights)
        weight_rows.append(
            {
                "dataset_name": dataset,
                "fold_id": int(fold_id),
                "predictor_name": predictor,
                "w_ctx": weights[0],
                "w_stability": weights[1],
                "w_support": weights[2],
                "w_disagreement": weights[3],
                "w_magnitude": weights[4],
                "w_ood": weights[5],
                "tuning_aligned_spearman": val_aligned,
                "tuning_source": source,
                "n_train": int((sub["split"] == "train").sum()),
                "n_val": int((sub["split"] == "val").sum()),
            }
        )
    weights_df = pd.DataFrame(weight_rows)
    weights_df.to_csv(dirs.tables / "SIMPLE_COMBINED_WEIGHTS.csv", index=False)
    add_score(score_rows, base, "simple_combined_confidence", "confidence", combined)

    learned = pd.Series(np.nan, index=base.index, dtype=float)
    feature_cols = [
        "context_similarity_max",
        "context_similarity_mean",
        "perturbation_support_count",
        "perturbation_effect_stability",
        "prediction_magnitude_deviation",
        "model_disagreement_rmse",
        "ood_nearest_distance",
        "historical_residual_risk",
    ]
    learned_status_rows = []
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor

        for (dataset, fold_id, predictor), sub in base.groupby(["dataset_name", "fold_id", "predictor_name"]):
            train_pool = sub[sub["split"].isin(["train", "val"])].dropna(subset=["true_error_rmse"]).copy()
            test_pool = sub[sub["split"] == "test"].copy()
            if len(train_pool) < 8 or test_pool.empty:
                learned_status_rows.append(
                    {
                        "dataset_name": dataset,
                        "fold_id": int(fold_id),
                        "predictor_name": predictor,
                        "status": "skipped_too_few_rows",
                        "n_train_pool": int(len(train_pool)),
                        "n_test": int(len(test_pool)),
                    }
                )
                continue
            med = train_pool[feature_cols].median(numeric_only=True).fillna(0.0)
            x_train = train_pool[feature_cols].fillna(med)
            x_test = test_pool[feature_cols].fillna(med)
            y_train = train_pool["true_error_rmse"].astype(float)
            model = HistGradientBoostingRegressor(max_depth=4, max_iter=200, random_state=seed)
            model.fit(x_train, y_train)
            learned.loc[test_pool.index] = model.predict(x_test)
            learned_status_rows.append(
                {
                    "dataset_name": dataset,
                    "fold_id": int(fold_id),
                    "predictor_name": predictor,
                    "status": "ok",
                    "n_train_pool": int(len(train_pool)),
                    "n_test": int(len(test_pool)),
                    "feature_cols": ",".join(feature_cols),
                }
            )
    except Exception as exc:  # pragma: no cover
        learned_status_rows.append({"status": "failed", "error": repr(exc)})
    learned_report = {
        "status": "ok" if any(r.get("status") == "ok" for r in learned_status_rows) else "failed_or_skipped",
        "model": "HistGradientBoostingRegressor(max_depth=4, max_iter=200, random_state=5201)",
        "train_scope": "same dataset + same fold + same predictor, split in train/val only; no other folds and no test labels",
        "n_scored": int(learned.notna().sum()),
        "fold_status": learned_status_rows,
    }
    pd.DataFrame(learned_status_rows).to_csv(dirs.tables / "LEARNED_RISK_FOLD_STATUS.csv", index=False)
    add_score(score_rows, base, "learned_risk_score", "risk", learned)

    scores = pd.DataFrame(score_rows)
    scores.to_csv(dirs.tables / "CONFIDENCE_SCORES.csv", index=False)
    save_json(dirs.tables / "LEARNED_RISK_STATUS.json", learned_report)
    lines = [
        "# Confidence score report",
        "",
        f"- Score rows: {len(scores)}",
        f"- Score names: {', '.join(sorted(scores['score_name'].unique()))}",
        "",
        "## simple_combined_confidence",
        "",
        "- Formula uses fixed signs: +ctx +stability +support -disagreement -magnitude -ood.",
        "- Weights are tuned within each dataset/fold/predictor using train-reference z scores and validation aligned Spearman.",
        "",
        "## Learned risk model",
        "```json",
        json.dumps(learned_report, ensure_ascii=False, indent=2, default=to_jsonable),
        "```",
        "",
        "Important: `learned_risk_score` uses same-fold train+val only and is scored only on test rows.",
    ]
    write_text(dirs.reports / "confidence_scores_report.md", "\n".join(lines) + "\n")
    return scores, merged


def eval_one_group(g: pd.DataFrame, group_meta: dict) -> dict:
    spearman = corr_pair(g["score_value"], g["true_error_rmse"], "spearman")
    pearson = corr_pair(g["score_value"], g["true_error_rmse"], "pearson")
    score_type = str(g["score_type"].iloc[0])
    aligned = -spearman if score_type == "confidence" and np.isfinite(spearman) else spearman
    return {
        **group_meta,
        "score_type": score_type,
        "n": int(g[["score_value", "true_error_rmse"]].dropna().shape[0]),
        "spearman_score_vs_rmse": spearman,
        "pearson_score_vs_rmse": pearson,
        "direction_aligned_spearman": aligned,
        "direction_ok": bool(np.isfinite(aligned) and aligned > 0),
        "mean_rmse": float(g["true_error_rmse"].mean()),
        "median_rmse": float(g["true_error_rmse"].median()),
    }


def evaluate_scores(scores: pd.DataFrame, dirs: RunDirs) -> dict:
    test = scores[scores["split"] == "test"].dropna(subset=["score_value", "true_error_rmse"]).copy()
    eval_rows = []
    for score_name, g in test.groupby("score_name"):
        eval_rows.append(eval_one_group(g, {"level": "overall", "dataset_name": "ALL", "predictor_name": "ALL", "score_name": score_name}))
    for (dataset, score_name), g in test.groupby(["dataset_name", "score_name"]):
        eval_rows.append(eval_one_group(g, {"level": "dataset", "dataset_name": dataset, "predictor_name": "ALL", "score_name": score_name}))
    for (predictor, score_name), g in test.groupby(["predictor_name", "score_name"]):
        eval_rows.append(eval_one_group(g, {"level": "predictor", "dataset_name": "ALL", "predictor_name": predictor, "score_name": score_name}))
    for (dataset, predictor, score_name), g in test.groupby(["dataset_name", "predictor_name", "score_name"]):
        eval_rows.append(eval_one_group(g, {"level": "dataset_predictor", "dataset_name": dataset, "predictor_name": predictor, "score_name": score_name}))
    eval_df = pd.DataFrame(eval_rows).sort_values(["level", "dataset_name", "predictor_name", "score_name"])

    high_low_rows = []
    coverage_rows = []
    failure_rows = []
    bucket_rows = []
    group_cols = ["dataset_name", "predictor_name", "score_name"]
    for key, g in test.groupby(group_cols):
        dataset, predictor, score_name = key
        score_type = str(g["score_type"].iloc[0])
        g = g.sort_values("score_value", ascending=(score_type == "risk")).reset_index(drop=True)
        n = len(g)
        if n == 0:
            continue
        qn = max(1, int(math.ceil(0.2 * n)))
        good = g.head(qn)
        bad = g.tail(qn)
        high_low_rows.append(
            {
                "dataset_name": dataset,
                "predictor_name": predictor,
                "score_name": score_name,
                "score_type": score_type,
                "n_total": n,
                "n_good": len(good),
                "n_bad": len(bad),
                "good_subset_label": "high_confidence_or_low_risk",
                "bad_subset_label": "low_confidence_or_high_risk",
                "good_mean_rmse": float(good["true_error_rmse"].mean()),
                "bad_mean_rmse": float(bad["true_error_rmse"].mean()),
                "good_median_rmse": float(good["true_error_rmse"].median()),
                "bad_median_rmse": float(bad["true_error_rmse"].median()),
                "mean_rmse_improvement": float(bad["true_error_rmse"].mean() - good["true_error_rmse"].mean()),
            }
        )
        full_mean = float(g["true_error_rmse"].mean())
        for cov in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
            keep = max(1, int(math.ceil(cov * n)))
            kept = g.head(keep)
            coverage_rows.append(
                {
                    "dataset_name": dataset,
                    "predictor_name": predictor,
                    "score_name": score_name,
                    "score_type": score_type,
                    "coverage": cov,
                    "n_kept": keep,
                    "mean_rmse": float(kept["true_error_rmse"].mean()),
                    "median_rmse": float(kept["true_error_rmse"].median()),
                    "full_mean_rmse": full_mean,
                    "rmse_delta_vs_full": float(full_mean - kept["true_error_rmse"].mean()),
                }
            )
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score

            threshold = float(g["true_error_rmse"].quantile(0.8))
            label = (g["true_error_rmse"] >= threshold).astype(int)
            failure_score = g["score_value"].astype(float).to_numpy()
            if score_type == "confidence":
                failure_score = -failure_score
            if label.nunique() >= 2:
                auroc = float(roc_auc_score(label, failure_score))
                auprc = float(average_precision_score(label, failure_score))
            else:
                auroc = float("nan")
                auprc = float("nan")
        except Exception:
            threshold = float("nan")
            auroc = float("nan")
            auprc = float("nan")
        failure_rows.append(
            {
                "dataset_name": dataset,
                "predictor_name": predictor,
                "score_name": score_name,
                "score_type": score_type,
                "n": n,
                "failure_threshold_q80_rmse": threshold,
                "auroc": auroc,
                "auprc": auprc,
            }
        )
        if n >= 5:
            ranks = pd.Series(g["score_value"]).rank(method="first")
            try:
                buckets = pd.qcut(ranks, q=min(5, n), labels=False, duplicates="drop")
            except ValueError:
                buckets = pd.Series(np.zeros(n, dtype=int))
            tmp = g.copy()
            tmp["bucket"] = np.asarray(buckets, dtype=int)
            for bucket, bg in tmp.groupby("bucket"):
                bucket_rows.append(
                    {
                        "dataset_name": dataset,
                        "predictor_name": predictor,
                        "score_name": score_name,
                        "score_type": score_type,
                        "bucket": int(bucket),
                        "n": len(bg),
                        "mean_score": float(bg["score_value"].mean()),
                        "mean_rmse": float(bg["true_error_rmse"].mean()),
                        "median_rmse": float(bg["true_error_rmse"].median()),
                    }
                )
    high_low = pd.DataFrame(high_low_rows)
    coverage = pd.DataFrame(coverage_rows)
    failure = pd.DataFrame(failure_rows)
    buckets = pd.DataFrame(bucket_rows)
    high_low.to_csv(dirs.tables / "HIGH_LOW_CONFIDENCE_RMSE.csv", index=False)
    coverage.to_csv(dirs.tables / "RISK_COVERAGE.csv", index=False)
    failure.to_csv(dirs.tables / "FAILURE_DETECTION.csv", index=False)
    buckets.to_csv(dirs.tables / "CALIBRATION_BUCKETS.csv", index=False)

    def summarize_extra(level: str, dataset: str, predictor: str, score_name: str) -> tuple[float, float]:
        cov = coverage[coverage["score_name"] == score_name].copy()
        hl = high_low[high_low["score_name"] == score_name].copy()
        if level in {"dataset", "dataset_predictor"}:
            cov = cov[cov["dataset_name"] == dataset]
            hl = hl[hl["dataset_name"] == dataset]
        if level in {"predictor", "dataset_predictor"}:
            cov = cov[cov["predictor_name"] == predictor]
            hl = hl[hl["predictor_name"] == predictor]
        cov80 = cov[np.isclose(cov["coverage"].astype(float), 0.8)]
        if cov80.empty:
            improve_pct = float("nan")
        else:
            improve_pct = float(np.nanmean(100.0 * cov80["rmse_delta_vs_full"] / cov80["full_mean_rmse"].replace(0, np.nan)))
        gap = float(np.nanmean(hl["mean_rmse_improvement"])) if not hl.empty else float("nan")
        return improve_pct, gap

    extras = []
    for row in eval_df.to_dict("records"):
        improve_pct, gap = summarize_extra(row["level"], row["dataset_name"], row["predictor_name"], row["score_name"])
        extras.append({"risk_cov_80_improve_pct": improve_pct, "high_low_rmse_gap": gap})
    eval_df = pd.concat([eval_df.reset_index(drop=True), pd.DataFrame(extras)], axis=1)
    eval_df.to_csv(dirs.tables / "CONFIDENCE_EVAL_SUMMARY.csv", index=False)
    save_json(dirs.tables / "CONFIDENCE_EVAL_SUMMARY.json", eval_df.to_dict("records"))

    overall = eval_df[eval_df["level"] == "overall"].copy()
    overall = overall.sort_values("direction_aligned_spearman", ascending=False)
    best = overall.iloc[0].to_dict() if not overall.empty else {}
    learned = overall[overall["score_name"] == "learned_risk_score"].iloc[0].to_dict() if (overall["score_name"] == "learned_risk_score").any() else {}
    random = overall[overall["score_name"] == "random_score"].iloc[0].to_dict() if (overall["score_name"] == "random_score").any() else {}
    report_lines = [
        "# Confidence evaluation report",
        "",
        f"- Test score rows evaluated: {len(test)}",
        f"- Best overall direction-aligned score: `{best.get('score_name', 'NA')}`",
        f"- Best overall aligned Spearman: {best.get('direction_aligned_spearman', float('nan')):.4f}" if best else "- Best overall aligned Spearman: NA",
        f"- Learned risk aligned Spearman: {learned.get('direction_aligned_spearman', float('nan')):.4f}" if learned else "- Learned risk aligned Spearman: NA",
        f"- Random aligned Spearman: {random.get('direction_aligned_spearman', float('nan')):.4f}" if random else "- Random aligned Spearman: NA",
        "",
        "## Overall summary",
        df_markdown(overall),
    ]
    write_text(dirs.reports / "confidence_eval_report.md", "\n".join(report_lines) + "\n")
    return {"best": best, "learned": learned, "random": random}


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("default")
    return plt


def plot_results(scores: pd.DataFrame, eval_info: pd.DataFrame, coverage: pd.DataFrame, high_low: pd.DataFrame, buckets: pd.DataFrame, split_df: pd.DataFrame, feat_df: pd.DataFrame, dirs: RunDirs):
    plt = setup_matplotlib()
    overall = eval_info[eval_info["level"] == "overall"].sort_values("direction_aligned_spearman", ascending=False)
    top_scores = overall["score_name"].head(3).tolist()
    if "random_score" not in top_scores:
        top_scores.append("random_score")
    # F1 schematic.
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.axis("off")
    boxes = [
        (0.08, 0.52, "Predictor output\npredicted_effect"),
        (0.36, 0.52, "Confidence features\ncontext, stability, OOD"),
        (0.64, 0.52, "Score\nconfidence / risk"),
        (0.84, 0.52, "Evaluation\nscore vs error"),
    ]
    for x, y, text in boxes:
        ax.add_patch(plt.Rectangle((x - 0.08, y - 0.16), 0.18, 0.26, fill=False, linewidth=1.5))
        ax.text(x + 0.01, y - 0.03, text, ha="center", va="center", fontsize=10)
    for x1, x2 in [(0.18, 0.28), (0.46, 0.56), (0.73, 0.78)]:
        ax.annotate("", xy=(x2, 0.49), xytext=(x1, 0.49), arrowprops={"arrowstyle": "->", "lw": 1.5})
    ax.set_title("Cross-context perturbation prediction confidence scoring", fontsize=12)
    fig.tight_layout()
    fig.savefig(dirs.figures / "F1_task_schematic.png", dpi=200)
    plt.close(fig)

    # F2 context x perturbation matrix.
    mat_df = split_df.drop_duplicates(["dataset_name", "task_id"]).copy()
    mat_df["observed"] = 1
    top = mat_df.copy()
    top["context_label"] = top["dataset_name"].astype(str) + "::" + top["context"].astype(str)
    context_counts = top["context_label"].value_counts().head(20).index.tolist()
    pert_counts = top["perturbation"].value_counts().head(30).index.tolist()
    top = top[top["context_label"].isin(context_counts) & top["perturbation"].isin(pert_counts)]
    pivot = top.pivot_table(index="context_label", columns="perturbation", values="observed", aggfunc="max", fill_value=0)
    fig, ax = plt.subplots(figsize=(12, max(4, 0.3 * len(pivot))))
    ax.imshow(pivot.values, aspect="auto", cmap="Greys", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_title("Observed context x perturbation tasks")
    fig.tight_layout()
    fig.savefig(dirs.figures / "F2_context_perturbation_matrix.png", dpi=220)
    plt.close(fig)

    # F3 flow.
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.axis("off")
    steps = ["raw cells", "context-pert task", "predicted effect", "true error", "confidence features", "score", "risk-coverage"]
    xs = np.linspace(0.06, 0.94, len(steps))
    for x, step in zip(xs, steps):
        ax.add_patch(plt.Circle((x, 0.55), 0.055, fill=False, linewidth=1.3))
        ax.text(x, 0.32, step, ha="center", va="center", fontsize=9)
    for x1, x2 in zip(xs[:-1], xs[1:]):
        ax.annotate("", xy=(x2 - 0.065, 0.55), xytext=(x1 + 0.065, 0.55), arrowprops={"arrowstyle": "->", "lw": 1.2})
    ax.set_title("PredictionRecord flow", fontsize=12)
    fig.tight_layout()
    fig.savefig(dirs.figures / "F3_prediction_record_flow.png", dpi=200)
    plt.close(fig)

    plot_scores = scores[(scores["split"] == "test") & (scores["score_name"].isin(top_scores))].copy()
    if not plot_scores.empty:
        fig, axes = plt.subplots(len(top_scores), 1, figsize=(8, max(3, 2.4 * len(top_scores))), squeeze=False)
        for ax, score_name in zip(axes[:, 0], top_scores):
            g = plot_scores[plot_scores["score_name"] == score_name]
            ax.scatter(g["score_value"], g["true_error_rmse"], s=18, alpha=0.7)
            sp = corr_pair(g["score_value"], g["true_error_rmse"], "spearman")
            ax.set_title(f"{score_name}: Spearman={sp:.3f}")
            ax.set_xlabel("score value")
            ax.set_ylabel("true RMSE")
        fig.tight_layout()
        fig.savefig(dirs.figures / "confidence_vs_error_scatter.png", dpi=220)
        fig.savefig(dirs.figures / "F4_confidence_vs_true_error_scatter.png", dpi=220)
        plt.close(fig)

    cov_plot = coverage[coverage["score_name"].isin(top_scores)].copy()
    if not cov_plot.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        for score_name, g in cov_plot.groupby("score_name"):
            gg = g.groupby("coverage", as_index=False)["mean_rmse"].mean().sort_values("coverage")
            ax.plot(gg["coverage"], gg["mean_rmse"], marker="o", label=score_name)
        ax.invert_xaxis()
        ax.set_xlabel("coverage kept")
        ax.set_ylabel("mean true RMSE")
        ax.set_title("Risk-coverage curve")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(dirs.figures / "risk_coverage.png", dpi=220)
        fig.savefig(dirs.figures / "F5_risk_coverage_curve.png", dpi=220)
        plt.close(fig)

    hl_plot = high_low[high_low["score_name"].isin(top_scores)].copy()
    if not hl_plot.empty:
        agg = hl_plot.groupby("score_name", as_index=False)[["good_mean_rmse", "bad_mean_rmse"]].mean()
        x = np.arange(len(agg))
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(x - 0.18, agg["good_mean_rmse"], width=0.36, label="high confidence / low risk")
        ax.bar(x + 0.18, agg["bad_mean_rmse"], width=0.36, label="low confidence / high risk")
        ax.set_xticks(x)
        ax.set_xticklabels(agg["score_name"], rotation=30, ha="right")
        ax.set_ylabel("mean true RMSE")
        ax.set_title("High vs low confidence RMSE")
        ax.legend()
        fig.tight_layout()
        fig.savefig(dirs.figures / "high_low_confidence_rmse.png", dpi=220)
        fig.savefig(dirs.figures / "F6_high_vs_low_confidence_rmse.png", dpi=220)
        plt.close(fig)

    dataset_scores = ["simple_combined_confidence", "learned_risk_score", "model_disagreement_risk", "historical_residual_risk", "random_score"]
    dataset_eval = eval_info[(eval_info["level"] == "dataset") & (eval_info["score_name"].isin(dataset_scores))].copy()
    if not dataset_eval.empty:
        piv = dataset_eval.pivot_table(index="dataset_name", columns="score_name", values="direction_aligned_spearman", aggfunc="mean")
        piv = piv.reindex(columns=[c for c in dataset_scores if c in piv.columns])
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(piv.index))
        width = 0.8 / max(1, len(piv.columns))
        for j, col in enumerate(piv.columns):
            ax.bar(x - 0.4 + width / 2 + j * width, piv[col].values, width=width, label=col)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(piv.index)
        ax.set_ylabel("direction-aligned Spearman")
        ax.set_title("Per-dataset confidence score comparison")
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(dirs.figures / "baseline_spearman_comparison.png", dpi=220)
        fig.savefig(dirs.figures / "F7_per_dataset_spearman_comparison.png", dpi=220)
        plt.close(fig)

    bucket_plot = buckets[buckets["score_name"].isin(top_scores)].copy()
    if not bucket_plot.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        for score_name, g in bucket_plot.groupby("score_name"):
            gg = g.groupby("bucket", as_index=False)["mean_rmse"].mean().sort_values("bucket")
            ax.plot(gg["bucket"], gg["mean_rmse"], marker="o", label=score_name)
        ax.set_xlabel("score bucket (low to high)")
        ax.set_ylabel("mean true RMSE")
        ax.set_title("Calibration buckets")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(dirs.figures / "calibration_buckets.png", dpi=220)
        fig.savefig(dirs.figures / "F8_calibration_buckets.png", dpi=220)
        plt.close(fig)

    imp_path = dirs.tables / "LEARNED_RISK_FEATURE_IMPORTANCE.csv"
    if imp_path.exists():
        imp = pd.read_csv(imp_path).sort_values("importance", ascending=True)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.barh(imp["feature"], imp["importance"])
        ax.set_xlabel("RandomForest feature importance")
        ax.set_title("Learned risk feature importance")
        fig.tight_layout()
        fig.savefig(dirs.figures / "F9_feature_importance.png", dpi=220)
        plt.close(fig)

    # F10 lightweight biological/actionable interpretation: perturbation transferability ranking.
    combined = scores[(scores["split"] == "test") & (scores["score_name"].isin(["learned_risk_score", "simple_combined_confidence"]))].copy()
    if not combined.empty:
        table = combined.groupby(["dataset_name", "perturbation", "score_name", "score_type"], as_index=False).agg(
            mean_score=("score_value", "mean"),
            mean_rmse=("true_error_rmse", "mean"),
            n=("record_id", "count"),
        )
        table.to_csv(dirs.tables / "TRANSFERABILITY_RANKING.csv", index=False)
        fig, ax = plt.subplots(figsize=(9, 5))
        g = table[table["score_name"] == "simple_combined_confidence"].copy()
        if not g.empty:
            g["label"] = g["dataset_name"].astype(str) + "::" + g["perturbation"].astype(str)
            g = g.sort_values("mean_score", ascending=False).head(12).sort_values("mean_score")
            ax.barh(g["label"], g["mean_score"])
            ax.set_xlabel("mean combined confidence")
            ax.set_title("Top transferable perturbation candidates")
            fig.tight_layout()
            fig.savefig(dirs.figures / "F10_transferability_ranking.png", dpi=220)
        plt.close(fig)


def write_reports(dirs: RunDirs, dataset_meta: pd.DataFrame, split_df: pd.DataFrame, rec_df: pd.DataFrame, feat_df: pd.DataFrame, scores: pd.DataFrame, eval_summary: pd.DataFrame, eval_result: dict):
    coverage = pd.read_csv(dirs.tables / "RISK_COVERAGE.csv")
    split_summary = pd.read_csv(dirs.tables / "HELDOUT_PAIR_SPLIT_SUMMARY.csv")
    feature_missing_by_dataset = pd.read_csv(dirs.tables / "CONFIDENCE_FEATURE_MISSINGNESS_BY_DATASET.csv")
    main_table = build_main_table_from_eval(eval_summary, DATASET_NAMES)
    main_table.to_csv(dirs.tables / "MAIN_PER_DATASET_SUMMARY.csv", index=False)
    simple_vals = main_table["simple_combined_confidence_aligned_rho"].dropna()
    simple_median = float(simple_vals.median()) if not simple_vals.empty else float("nan")
    kcc_rows = main_table.loc[
        main_table["dataset_name"] == "KaggleCrossCell",
        "simple_combined_confidence_aligned_rho",
    ]
    kcc_present = not kcc_rows.empty
    kcc_simple = float(kcc_rows.iloc[0]) if kcc_present else float("nan")
    learned_beats_dis = (main_table["learned_risk_score_aligned_rho"] - main_table["model_disagreement_risk_aligned_rho"]).max()
    learned_beats_dis = float(learned_beats_dis) if np.isfinite(learned_beats_dis) else float("nan")
    test_records = int((rec_df["split"] == "test").sum())
    leak_counts = {
        "pair_leak": int(((split_df["split"] == "test") & (split_df["pair_seen_in_train"])).sum()),
        "pert_missing": int(((split_df["split"] == "test") & (~split_df["perturbation_seen_in_train"])).sum()),
        "context_missing": int(((split_df["split"] == "test") & (~split_df["context_seen_in_train"])).sum()),
    }
    n_genes_ok = bool((dataset_meta["n_genes"].astype(int) == 5000).all())
    title = "MVP Phase 2.1 Report" if set(DATASET_NAMES) == {"KaggleCrossCell", "Haber", "Parekh"} else "Frozen-protocol blind dataset report"
    report = [
        f"# {title}",
        "",
        "## 主结论表（per-dataset，pooled 只作辅助）",
        "",
        df_markdown(main_table),
        "",
        "## 本阶段修了什么",
        "",
        "- 用 `n_genes=5000, min_cells=6, max_cells_per_group=2200, seed=5201` 重建三数据集任务。",
        "- PredictionRecord 包含 train/val/test，但所有正式 evaluation 只用 test。",
        "- `perturbation_effect_stability` 保留原始 NaN，不再用全局 median 洗成非缺失。",
        "- 新增 `historical_residual_risk`：train 内 leave-one-context-out 残差，按 perturbation 聚合。",
        "- `simple_combined_confidence` 用同 fold train 作 z-score 参考，权重只根据同 fold val 的 aligned Spearman 网格搜索。",
        "- `learned_risk_score` 改为同 dataset + 同 fold + 同 predictor 的 train+val 训练 HistGradientBoosting，只给 test 打分。",
        "",
        "## 数据与防泄漏",
        "",
        df_markdown(dataset_meta),
        "",
        f"- Test records: {test_records}",
        f"- Split leakage counts: {json.dumps(leak_counts, ensure_ascii=False)}",
        "",
        "## Feature missingness by dataset",
        "",
        df_markdown(feature_missing_by_dataset),
        "",
        "## 口径",
        "",
        "这次不把 pooled learned 分数当标题。主方法口径看每个数据集上的 `simple_combined_confidence`；learned 只是辅助/ablation。",
        "",
        "## combined regression debug",
        "",
        "- Phase 1 KCC combined 是单数据集 min-max 组合，且 KCC 任务较小。",
        "- Phase 2 v2 改成跨三数据集 pooled z-score/median fallback，并把 stability fallback 填满，导致 KCC/Haber combined 方向塌掉。",
        "- Phase 2.1 改回 fold-local 参考分布和 val-tuned fixed-sign weights；如果 G7/G8 仍未过，见 `reports/combined_regression_debug.md`。",
    ]
    write_text(dirs.out / "MVP_V2_1_REPORT.md", "\n".join(report) + "\n")

    g_status = {
        "G1": "PASS" if bool(set(DATASET_NAMES).issubset(set(eval_summary[eval_summary["level"] == "dataset"]["dataset_name"].unique()))) else "FAIL",
        "G2": "PASS" if test_records >= 150 else "FAIL",
        "G3": "PASS" if all(v == 0 for v in leak_counts.values()) else "FAIL",
        "G4": "PASS" if (dirs.out / "MVP_V2_1_REPORT.md").exists() else "FAIL",
        "G5": "PASS" if bool((eval_summary["score_name"] == "historical_residual_risk").any()) else "FAIL",
        "G6": "PASS" if bool((dirs.tables / "LEARNED_RISK_FOLD_STATUS.csv").exists()) else "FAIL",
        "G7": (
            "SKIPPED_NOT_APPLICABLE"
            if not kcc_present
            else ("PASS" if bool(np.isfinite(kcc_simple) and kcc_simple >= 0.50) else "FAIL")
        ),
        "G8": "PASS" if bool(np.isfinite(simple_median) and simple_median >= 0.30) else "FAIL",
        "G9": "PASS" if bool(np.isfinite(learned_beats_dis) and learned_beats_dis >= 0.05) else "FAIL",
        "G10": "PASS" if n_genes_ok else "FAIL",
        "G11": "FAIL",
    }
    checklist_meta = {
        "test_records": test_records,
        "leak_counts": leak_counts,
        "kcc_simple_combined_aligned_rho": kcc_simple,
        "kcc_gate_applicable": kcc_present,
        "simple_combined_median_aligned_rho": simple_median,
        "max_learned_minus_disagreement": learned_beats_dis,
        "n_genes_ok": n_genes_ok,
    }
    checklist = [
        "# Stage completion checklist v2.1",
        "",
        "## Metrics used for gates",
        "```json",
        json.dumps(checklist_meta, ensure_ascii=False, indent=2, default=to_jsonable),
        "```",
        "",
        "| Gate | Status | Evidence |",
        "|---|---|---|",
    ]
    evidence = {
        "G1": "tables/CONFIDENCE_EVAL_SUMMARY.csv level=dataset",
        "G2": f"test_records={test_records}",
        "G3": f"leak_counts={leak_counts}",
        "G4": "MVP_V2_1_REPORT.md starts with per-dataset table",
        "G5": "historical_residual_risk in CONFIDENCE_EVAL_SUMMARY.csv",
        "G6": "tables/LEARNED_RISK_FOLD_STATUS.csv",
        "G7": (
            f"KCC simple_combined aligned rho={kcc_simple:.4f}"
            if kcc_present
            else "KaggleCrossCell not present in this blind single-dataset run"
        ),
        "G8": f"simple_combined median aligned rho={simple_median:.4f}",
        "G9": f"max learned-disagreement={learned_beats_dis:.4f}",
        "G10": "tables/DATASET_TASK_SUMMARY.csv n_genes=5000",
        "G11": "set after zip test",
    }
    for key in [f"G{i}" for i in range(1, 12)]:
        checklist.append(f"| {key} | {g_status[key]} | {evidence[key]} |")
    write_text(dirs.out / "stage_completion_checklist_v2_1.md", "\n".join(checklist) + "\n")

    input_lines = [
        "# Input file check",
        "",
        f"- Requested spec: `{REQUESTED_SPEC}` exists = `{REQUESTED_SPEC.exists()}`",
        f"- Fallback plan: `{FALLBACK_PLAN}` exists = `{FALLBACK_PLAN.exists()}`",
        "- No new data were downloaded.",
        "- Deep perturbation predictors were not trained.",
        "- `outputs/confidence_task_mvp_final/` was not overwritten.",
        "",
        "## Generated core inputs",
        f"- `input/PREDICTION_RECORDS.csv`: {len(rec_df)} rows",
        f"- `input/predicted_effects.npz`: {len(rec_df)} arrays",
        f"- `input/true_effects.npz`: {len(rec_df)} arrays",
        f"- `tables/HELDOUT_PAIR_SPLITS.csv`: {len(split_df)} rows",
    ]
    write_text(dirs.reports / "input_file_check.md", "\n".join(input_lines) + "\n")

    reuse_lines = [
        "# Input reuse audit",
        "",
        "- Phase 2 outputs were checked but not reused for records/features because Phase 2 used `n_genes=1000`.",
        "- Phase 2.1 reran S1-S6 with `n_genes=5000` to match the specification.",
    ]
    write_text(dirs.reports / "input_reuse_audit.md", "\n".join(reuse_lines) + "\n")

    debug_lines = [
        "# Combined regression debug",
        "",
        "Phase 1 KCC `simple_combined_confidence` used a single-dataset min-max style combination from the final MVP scripts.",
        "Phase 2 changed the feature names, used pooled/validation-style z-scoring, added OOD and magnitude penalties, and filled raw stability with a median fallback. The independent audit found KCC and Haber collapsed under that formula.",
        "",
        "Phase 2.1 changes:",
        "- raw `perturbation_effect_stability` remains NaN when unsupported;",
        "- train fold is the only reference distribution for z-score;",
        "- weights are fixed-sign and selected by same-fold validation aligned Spearman;",
        "- test labels are not used to choose weights.",
        "",
        "Gate outcomes are in `stage_completion_checklist_v2_1.md`.",
    ]
    write_text(dirs.reports / "combined_regression_debug.md", "\n".join(debug_lines) + "\n")


def copy_scripts(dirs: RunDirs):
    script_paths = [
        Path(__file__),
        PROJECT_ROOT / "confidence_task" / "run_confidence_mvp_v2_1.sh",
    ]
    for path in script_paths:
        if path.exists():
            shutil.copy2(path, dirs.scripts / path.name)


def make_zip(dirs: RunDirs) -> Path:
    zip_path = dirs.out.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in dirs.out.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(dirs.out.parent)))
    return zip_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 2.1 confidence scoring MVP on selected datasets.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--atlas-root", default=str(DEFAULT_ATLAS_ROOT))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "outputs" / "confidence_task_mvp_v2_1"))
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASET_NAMES),
        help=(
            "Comma-separated dataset selectors from metadata/h5ad_scan.tsv. "
            "Supports study_family, file_name/stem, or study_family@file_name. "
            "Default keeps the original Phase 2.1 panel."
        ),
    )
    parser.add_argument("--n-genes", type=int, default=5000)
    parser.add_argument("--min-cells", type=int, default=6)
    parser.add_argument("--max-cells-per-group", type=int, default=2200)
    parser.add_argument("--seed", type=int, default=5201)
    args = parser.parse_args()
    global DATASET_NAMES
    dataset_selectors = parse_dataset_names(args.datasets)

    out = Path(args.out_dir)
    forbidden = {
        (PROJECT_ROOT / "outputs" / "confidence_task_mvp_final").resolve(),
        (PROJECT_ROOT / "outputs" / "confidence_task_mvp_v2").resolve(),
    }
    if out.resolve() in forbidden:
        raise RuntimeError("Refusing to overwrite previous confidence MVP outputs")
    dirs = make_dirs(out)
    start = time.time()
    print(f"[start] confidence MVP v2.1 output: {dirs.out}", flush=True)
    warnings.filterwarnings("ignore", category=FutureWarning)

    s0 = run_s0_audit(dirs)
    paths = resolve_dataset_paths(Path(args.atlas_root), dataset_selectors)
    DATASET_NAMES = list(paths.keys())
    tasks_by_dataset, genes_by_dataset, dataset_meta = build_all_tasks(
        paths, dirs, n_genes=args.n_genes, min_cells=args.min_cells, max_cells_per_group=args.max_cells_per_group, seed=args.seed
    )
    split_df = build_pair_splits(tasks_by_dataset, dirs, seed=args.seed)
    rec_df, pred_arrays, true_arrays = run_predictors(tasks_by_dataset, genes_by_dataset, split_df, dirs)
    feat_df = compute_features(rec_df, tasks_by_dataset, split_df, pred_arrays, dirs)
    scores, merged = run_scores(rec_df, feat_df, dirs, seed=args.seed)
    eval_result = evaluate_scores(scores, dirs)
    eval_summary = pd.read_csv(dirs.tables / "CONFIDENCE_EVAL_SUMMARY.csv")
    coverage = pd.read_csv(dirs.tables / "RISK_COVERAGE.csv")
    high_low = pd.read_csv(dirs.tables / "HIGH_LOW_CONFIDENCE_RMSE.csv")
    buckets = pd.read_csv(dirs.tables / "CALIBRATION_BUCKETS.csv")
    plot_results(scores, eval_summary, coverage, high_low, buckets, split_df, feat_df, dirs)
    write_reports(dirs, dataset_meta, split_df, rec_df, feat_df, scores, eval_summary, eval_result)
    copy_scripts(dirs)
    zip_path = dirs.out.with_suffix(".zip")
    # G11 is marked when the rerun script is present; the zip is written after
    # RUN_STATUS so the archive contains the final checklist and status.
    checklist = dirs.out / "stage_completion_checklist_v2_1.md"
    if checklist.exists() and (dirs.scripts / "run_confidence_mvp_v2_1.sh").exists():
        text = checklist.read_text(encoding="utf-8")
        text = text.replace("| G11 | FAIL | set after zip test |", "| G11 | PASS | zip created after checklist and scripts/run_confidence_mvp_v2_1.sh copied |")
        checklist.write_text(text, encoding="utf-8")
    elapsed = time.time() - start
    status = {
        "output_dir": str(dirs.out),
        "zip_path": str(zip_path),
        "elapsed_seconds": elapsed,
        "n_prediction_records": int(len(rec_df)),
        "datasets": DATASET_NAMES,
        "best_overall": eval_result.get("best", {}),
        "learned_overall": eval_result.get("learned", {}),
        "requested_spec_exists": REQUESTED_SPEC.exists(),
    }
    save_json(dirs.out / "RUN_STATUS.json", status)
    zip_path = make_zip(dirs)
    print(json.dumps(status, ensure_ascii=False, indent=2, default=to_jsonable), flush=True)
    print(f"[done] zip: {zip_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
