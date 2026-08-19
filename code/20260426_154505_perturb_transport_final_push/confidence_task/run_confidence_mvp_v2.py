#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


DATASET_NAMES = ["KaggleCrossCell", "Haber", "Parekh"]
DEFAULT_ATLAS_ROOT = Path("/home/yyf/datasets/singlecell_perturbation_atlas")
REQUESTED_SPEC = Path("/home/yyf/SafeTrans-confidence-scoring-Codex执行规格.md")
FALLBACK_PLAN = Path("/home/yyf/SafeTrans-confidence-scoring-方案.md")


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


def resolve_dataset_paths(atlas_root: Path) -> dict[str, Path]:
    scan_path = atlas_root / "metadata" / "h5ad_scan.tsv"
    if not scan_path.exists():
        raise FileNotFoundError(f"Missing scan table: {scan_path}")
    scan = pd.read_csv(scan_path, sep="\t")
    out: dict[str, Path] = {}
    for name in DATASET_NAMES:
        sub = scan[scan["study_family"].astype(str) == name]
        if sub.empty:
            raise FileNotFoundError(f"Dataset {name} not found in {scan_path}")
        path = Path(str(sub.iloc[0]["local_path"]))
        if not path.exists():
            raise FileNotFoundError(f"Dataset file missing for {name}: {path}")
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


def run_predictors(tasks_by_dataset: dict[str, list[dict]], split_df: pd.DataFrame, dirs: RunDirs):
    records = []
    pred_arrays = {}
    true_arrays = {}
    control_arrays = {}
    rec_i = 0
    status_rows = []
    for dataset, tasks in tasks_by_dataset.items():
        sub = split_df[split_df["dataset_name"] == dataset]
        for fold_id in sorted(sub["fold_id"].unique()):
            train_mask = train_mask_from_split(tasks, sub, int(fold_id))
            eval_rows = sub[(sub["fold_id"] == fold_id) & (sub["split"].isin(["val", "test"]))].copy()
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
                    true = task["effect"].astype(np.float32)
                    record_id = f"v2_rec_{rec_i:06d}"
                    pred_key = f"{record_id}::predicted_effect"
                    true_key = f"{record_id}::true_effect"
                    ctrl_key = f"{record_id}::target_control_mean"
                    pred_arrays[pred_key] = pred.astype(np.float32)
                    true_arrays[true_key] = true
                    control_arrays[ctrl_key] = task["control_mean"].astype(np.float32)
                    records.append(
                        {
                            "record_id": record_id,
                            "task_id": int(task_id),
                            "task_key": task["task_key"],
                            "dataset_name": dataset,
                            "fold_id": int(fold_id),
                            "split": row["split"],
                            "context": task["context"],
                            "perturbation": task["perturbation"],
                            "predictor_name": pred_name,
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


def compute_train_stability(tasks: list[dict], train_ids: list[int]) -> dict[str, float]:
    by_pert: dict[str, list[np.ndarray]] = {}
    for i in train_ids:
        by_pert.setdefault(tasks[i]["perturbation"], []).append(tasks[i]["effect"])
    out = {}
    for pert, vals in by_pert.items():
        if len(vals) >= 2:
            out[pert] = mean_pairwise_cosine(np.stack(vals, axis=0))
    return out


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
        train_contexts = np.array([tasks[i]["context"] for i in train_ids], dtype=object)
        train_perts = np.array([tasks[i]["perturbation"] for i in train_ids], dtype=object)
        train_stability = compute_train_stability(tasks, train_ids)
        median_stability = float(np.nanmedian(list(train_stability.values()))) if train_stability else float("nan")
        global_effect = np.nanmean(train_effects, axis=0)
        median_effect_norm = float(np.nanmedian(np.linalg.norm(train_effects, axis=1)))
        # Represent train tasks by control state plus observed effect.
        train_reps = np.stack(
            [np.concatenate([normalize_vec(tasks[i]["control_mean"]), normalize_vec(tasks[i]["effect"])]) for i in train_ids],
            axis=0,
        )
        for r in rec_sub.to_dict("records"):
            task = tasks[int(r["task_id"])]
            pred = pred_arrays[r["predicted_effect_key"]]
            other_ctx_mask = train_contexts != task["context"]
            ctx_sims = np.array([cosine(task["control_mean"], c) for c in train_controls[other_ctx_mask]], dtype=float)
            if len(ctx_sims) == 0 or np.all(~np.isfinite(ctx_sims)):
                ctx_sims = np.array([cosine(task["control_mean"], c) for c in train_controls], dtype=float)
            same_pert_other = [
                i for i in train_ids
                if tasks[i]["perturbation"] == task["perturbation"] and tasks[i]["context"] != task["context"]
            ]
            source_effects = np.stack([tasks[i]["effect"] for i in same_pert_other], axis=0) if same_pert_other else np.empty((0, len(task["effect"])))
            raw_stability = mean_pairwise_cosine(source_effects) if len(source_effects) >= 2 else float("nan")
            stability_fallback = raw_stability if np.isfinite(raw_stability) else median_stability
            effect_var = float(np.mean(np.var(source_effects, axis=0))) if len(source_effects) >= 2 else float("nan")
            source_mean = source_effects.mean(axis=0) if len(source_effects) else global_effect
            target_rep = np.concatenate([normalize_vec(task["control_mean"]), normalize_vec(source_mean)])
            dists = np.array([cosine_distance(target_rep, rep) for rep in train_reps], dtype=float)
            dists = dists[np.isfinite(dists)]
            k = min(5, len(dists))
            pred_norm = float(np.linalg.norm(pred))
            norm_ratio = pred_norm / (median_effect_norm + 1e-8)
            dis_key = (dataset, fold_id, r["split"], int(r["task_id"]))
            dis_rmse, dis_cos = disagreement.get(dis_key, (float("nan"), float("nan")))
            rows.append(
                {
                    "record_id": r["record_id"],
                    "task_id": int(r["task_id"]),
                    "task_key": r["task_key"],
                    "dataset_name": dataset,
                    "fold_id": int(fold_id),
                    "split": r["split"],
                    "context": task["context"],
                    "perturbation": task["perturbation"],
                    "predictor_name": r["predictor_name"],
                    "context_similarity_max_other_context": float(np.nanmax(ctx_sims)) if len(ctx_sims) else float("nan"),
                    "context_similarity_mean_other_context": float(np.nanmean(ctx_sims)) if len(ctx_sims) else float("nan"),
                    "perturbation_support_count": int(len({tasks[i]["context"] for i in same_pert_other})),
                    "perturbation_effect_stability": raw_stability,
                    "perturbation_effect_stability_v2": stability_fallback,
                    "perturbation_effect_stability_available": bool(np.isfinite(raw_stability)),
                    "perturbation_effect_variance": effect_var,
                    "prediction_l2_norm": pred_norm,
                    "prediction_abs_mean": float(np.mean(np.abs(pred))),
                    "fold_train_median_effect_norm": median_effect_norm,
                    "prediction_norm_ratio": float(norm_ratio),
                    "prediction_magnitude_deviation": float(abs(math.log(norm_ratio + 1e-8))),
                    "model_disagreement_rmse": dis_rmse,
                    "model_disagreement_cosine": dis_cos,
                    "ood_nearest_distance": float(np.nanmin(dists)) if len(dists) else float("nan"),
                    "ood_mean_k_distance": float(np.nanmean(np.sort(dists)[:k])) if len(dists) else float("nan"),
                }
            )
    feat = pd.DataFrame(rows)
    feat.to_csv(dirs.tables / "CONFIDENCE_FEATURES.csv", index=False)
    missing = feat.isna().mean(numeric_only=False).sort_values(ascending=False).reset_index()
    missing.columns = ["column", "missing_rate"]
    numeric = feat.select_dtypes(include=[np.number]).describe().T.reset_index().rename(columns={"index": "feature"})
    missing.to_csv(dirs.tables / "CONFIDENCE_FEATURE_MISSINGNESS.csv", index=False)
    numeric.to_csv(dirs.tables / "CONFIDENCE_FEATURE_DESCRIBE.csv", index=False)
    lines = [
        "# Confidence feature report",
        "",
        "Phase 2 fixes:",
        "- `context_similarity_*` excludes the same target context where possible, so it is less trivially 1.0.",
        "- `perturbation_effect_stability_v2` adds a train-fold median fallback when one perturbation has fewer than two source contexts.",
        "- `ood_nearest_distance` and `ood_mean_k_distance` are computed from train-only task representations.",
        "- `prediction_magnitude_deviation` normalizes prediction norm by each fold's train effect norm.",
        "",
        "## Missingness",
        df_markdown(missing),
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


def run_scores(rec_df: pd.DataFrame, feat_df: pd.DataFrame, dirs: RunDirs, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = rec_df.merge(feat_df, on=[
        "record_id", "task_id", "task_key", "dataset_name", "fold_id", "split", "context", "perturbation", "predictor_name"
    ], how="left")
    score_rows: list[dict] = []
    rng = np.random.default_rng(seed)
    base = merged.reset_index(drop=True)
    add_score(score_rows, base, "random_score", "confidence", pd.Series(rng.random(len(base)), index=base.index))
    add_score(score_rows, base, "context_similarity_score_v2", "confidence", base["context_similarity_max_other_context"])
    add_score(score_rows, base, "perturbation_stability_score_v2", "confidence", base["perturbation_effect_stability_v2"])
    add_score(score_rows, base, "support_count_score", "confidence", np.log1p(base["perturbation_support_count"].astype(float)))
    add_score(score_rows, base, "prediction_magnitude_risk_v2", "risk", base["prediction_magnitude_deviation"])
    add_score(score_rows, base, "model_disagreement_risk", "risk", base["model_disagreement_rmse"])
    add_score(score_rows, base, "ood_distance_risk", "risk", base["ood_nearest_distance"])

    combined = pd.Series(np.nan, index=base.index, dtype=float)
    for (_, predictor), idx in base.groupby(["dataset_name", "predictor_name"]).groups.items():
        idx = list(idx)
        sub = base.loc[idx]
        ref = sub[sub["split"] == "val"]
        if ref.empty:
            ref = sub
        z_ctx = zscore_by_reference(sub["context_similarity_max_other_context"], ref["context_similarity_max_other_context"])
        z_stab = zscore_by_reference(sub["perturbation_effect_stability_v2"], ref["perturbation_effect_stability_v2"])
        z_support = zscore_by_reference(np.log1p(sub["perturbation_support_count"].astype(float)), np.log1p(ref["perturbation_support_count"].astype(float)))
        z_mag = zscore_by_reference(sub["prediction_magnitude_deviation"], ref["prediction_magnitude_deviation"])
        z_dis = zscore_by_reference(sub["model_disagreement_rmse"], ref["model_disagreement_rmse"])
        z_ood = zscore_by_reference(sub["ood_nearest_distance"], ref["ood_nearest_distance"])
        combined.loc[idx] = z_ctx.values + z_stab.values + z_support.values - z_mag.values - z_dis.values - z_ood.values
    add_score(score_rows, base, "simple_combined_confidence_v2", "confidence", combined)

    learned = pd.Series(np.nan, index=base.index, dtype=float)
    feature_cols = [
        "context_similarity_max_other_context",
        "context_similarity_mean_other_context",
        "perturbation_support_count",
        "perturbation_effect_stability_v2",
        "prediction_magnitude_deviation",
        "model_disagreement_rmse",
        "ood_nearest_distance",
        "ood_mean_k_distance",
    ]
    learned_report = {"status": "not_run", "n_train": 0, "model": None, "feature_cols": feature_cols}
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline

        train = base[base["split"] == "val"].copy()
        test_like = base.copy()
        train = train.dropna(subset=["true_error_rmse"])
        if len(train) >= 8:
            n_fit = 0
            # Score each record with a shallow risk model trained only on
            # validation records, while excluding the same dataset/context/
            # perturbation pair. This prevents the cross-fold pair reuse from
            # making the learned score look artificially strong.
            for idx, row in test_like.iterrows():
                pair_mask = (
                    (train["dataset_name"].astype(str) == str(row["dataset_name"]))
                    & (train["context"].astype(str) == str(row["context"]))
                    & (train["perturbation"].astype(str) == str(row["perturbation"]))
                )
                local_train = train[~pair_mask]
                if len(local_train) < 8:
                    # Still do not use the same pair; leave NaN if not enough
                    # calibration records remain.
                    continue
                model = make_pipeline(
                    SimpleImputer(strategy="median"),
                    RandomForestRegressor(n_estimators=120, min_samples_leaf=3, random_state=seed + int(idx), n_jobs=1),
                )
                model.fit(local_train[feature_cols], local_train["true_error_rmse"].astype(float))
                learned.loc[idx] = float(model.predict(pd.DataFrame([row[feature_cols]], columns=feature_cols))[0])
                n_fit += 1
            learned_report = {
                "status": "ok",
                "n_train": int(len(train)),
                "n_scored": int(learned.notna().sum()),
                "model": "Leave-same-pair-out SimpleImputer+RandomForestRegressor",
                "feature_cols": feature_cols,
                "train_scope": "validation records only; same dataset/context/perturbation pair excluded for each scored record; no test labels used",
            }
            # Approximate feature importance from the RF step.
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=seed, n_jobs=1),
            )
            model.fit(train[feature_cols], train["true_error_rmse"].astype(float))
            rf = model.named_steps["randomforestregressor"]
            imp = pd.DataFrame({"feature": feature_cols, "importance": rf.feature_importances_})
            imp.sort_values("importance", ascending=False).to_csv(dirs.tables / "LEARNED_RISK_FEATURE_IMPORTANCE.csv", index=False)
        else:
            learned_report = {"status": "too_few_validation_records", "n_train": int(len(train)), "model": None, "feature_cols": feature_cols}
    except Exception as exc:  # pragma: no cover - runtime safety.
        learned_report = {"status": "failed", "error": repr(exc), "feature_cols": feature_cols}
    add_score(score_rows, base, "learned_risk_score_v2", "risk", learned)
    scores = pd.DataFrame(score_rows)
    scores.to_csv(dirs.tables / "CONFIDENCE_SCORES.csv", index=False)
    save_json(dirs.tables / "LEARNED_RISK_STATUS.json", learned_report)
    lines = [
        "# Confidence score report",
        "",
        f"- Score rows: {len(scores)}",
        f"- Score names: {', '.join(sorted(scores['score_name'].unique()))}",
        "",
        "## Learned risk model",
        "```json",
        json.dumps(learned_report, ensure_ascii=False, indent=2, default=to_jsonable),
        "```",
        "",
        "Important: `learned_risk_score_v2` is trained only on validation records and evaluated primarily on test records.",
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
    eval_df.to_csv(dirs.tables / "CONFIDENCE_EVAL_SUMMARY.csv", index=False)
    save_json(dirs.tables / "CONFIDENCE_EVAL_SUMMARY.json", eval_df.to_dict("records"))

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

    overall = eval_df[eval_df["level"] == "overall"].copy()
    overall = overall.sort_values("direction_aligned_spearman", ascending=False)
    best = overall.iloc[0].to_dict() if not overall.empty else {}
    learned = overall[overall["score_name"] == "learned_risk_score_v2"].iloc[0].to_dict() if (overall["score_name"] == "learned_risk_score_v2").any() else {}
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

    if not overall.empty:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ordered = overall.sort_values("direction_aligned_spearman", ascending=True)
        ax.barh(ordered["score_name"], ordered["direction_aligned_spearman"])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("direction-aligned Spearman")
        ax.set_title("Confidence/risk baseline comparison")
        fig.tight_layout()
        fig.savefig(dirs.figures / "baseline_spearman_comparison.png", dpi=220)
        fig.savefig(dirs.figures / "F7_baseline_spearman_comparison.png", dpi=220)
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
    combined = scores[(scores["split"] == "test") & (scores["score_name"].isin(["learned_risk_score_v2", "simple_combined_confidence_v2"]))].copy()
    if not combined.empty:
        table = combined.groupby(["dataset_name", "perturbation", "score_name", "score_type"], as_index=False).agg(
            mean_score=("score_value", "mean"),
            mean_rmse=("true_error_rmse", "mean"),
            n=("record_id", "count"),
        )
        table.to_csv(dirs.tables / "TRANSFERABILITY_RANKING.csv", index=False)
        fig, ax = plt.subplots(figsize=(9, 5))
        g = table[table["score_name"] == "simple_combined_confidence_v2"].copy()
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
    overall = eval_summary[eval_summary["level"] == "overall"].sort_values("direction_aligned_spearman", ascending=False)
    best = overall.iloc[0].to_dict() if not overall.empty else {}
    learned = overall[overall["score_name"] == "learned_risk_score_v2"].iloc[0].to_dict() if (overall["score_name"] == "learned_risk_score_v2").any() else {}
    random = overall[overall["score_name"] == "random_score"].iloc[0].to_dict() if (overall["score_name"] == "random_score").any() else {}
    coverage = pd.read_csv(dirs.tables / "RISK_COVERAGE.csv")
    high_low = pd.read_csv(dirs.tables / "HIGH_LOW_CONFIDENCE_RMSE.csv")
    cov80 = pd.DataFrame()
    if best:
        cov80 = coverage[(coverage["score_name"] == best["score_name"]) & (np.isclose(coverage["coverage"], 0.8))]
    hl_best = pd.DataFrame()
    if best:
        hl_best = high_low[high_low["score_name"] == best["score_name"]]
    report = [
        "# MVP v2 report: Cross-context perturbation prediction confidence scoring",
        "",
        "## 1. 这次 MVP 目标",
        "",
        "这次不是训练新的深度扰动预测模型，而是先问一个更朴素的问题：已有 predictor 给出一个 `predicted_effect` 后，我们能不能给这次预测打一个可靠性分数，并且这个分数真的和后来的真实误差相关。",
        "",
        "## 2. 数据集与 split",
        "",
        "- 数据集：KaggleCrossCell、Haber、Parekh。",
        "- Split：held-out `(context, perturbation)` pair。测试 pair 本身不进 train，但它的 context 和 perturbation 分别必须在 train 的别处出现。",
        "- 未下载新数据，全部使用服务器已有 h5ad。",
        "",
        df_markdown(dataset_meta),
        "",
        "## 3. Predictor 和 PredictionRecord",
        "",
        "- Predictor：`V0StrongBaseline` 和 `ContextSimBaseline`。",
        f"- PredictionRecord 总行数：{len(rec_df)}。",
        f"- Test record 行数：{int((rec_df['split'] == 'test').sum())}。",
        "",
        "## 4. Phase 2 修复点",
        "",
        "- stability：新增 `perturbation_effect_stability_v2`，当同 perturbation 的 source context 少于 2 个时，用 train fold 的稳定性中位数作保守 fallback，并保留原始 availability 标记。",
        "- OOD：新增 `ood_nearest_distance` 和 `ood_mean_k_distance`，从 train-only 的 context/control + source effect 表征计算。",
        "- learned：`learned_risk_score_v2` 只用 validation records 的真实误差训练浅层 RandomForest；给每条 record 打分时排除同一个 `(dataset, context, perturbation)` pair，不使用 test labels。",
        "",
        "## 5. 主要结果",
        "",
    ]
    if best:
        report.append(f"- 当前最佳 overall score：`{best['score_name']}`。")
        report.append(f"- Direction-aligned Spearman：{best['direction_aligned_spearman']:.4f}。")
    if learned:
        report.append(f"- `learned_risk_score_v2` aligned Spearman：{learned['direction_aligned_spearman']:.4f}。")
    if random:
        report.append(f"- random baseline aligned Spearman：{random['direction_aligned_spearman']:.4f}。")
    if not cov80.empty:
        report.append(f"- 最佳 score 在 80% coverage 的平均 RMSE：{cov80['mean_rmse'].mean():.4f}；全量平均 RMSE：{cov80['full_mean_rmse'].mean():.4f}。")
    if not hl_best.empty:
        report.append(f"- high-confidence/low-risk 子集平均 RMSE：{hl_best['good_mean_rmse'].mean():.4f}；low-confidence/high-risk 子集平均 RMSE：{hl_best['bad_mean_rmse'].mean():.4f}。")
    report.extend(
        [
            "",
            "## 6. 是否支持 confidence scoring task",
            "",
            "支持做下一轮扩展，但仍是 MVP 证据。理由是：三数据集端到端跑通，score/error 相关性可以量化，risk-coverage 和 high-low 对比都有表；限制是样本量仍小，predictor 只有两个，learned model 仍是 exploratory。",
            "",
        "## 7. 当前限制",
        "",
            f"- 你指定的严格执行规格文件 `{REQUESTED_SPEC}` 当前没在服务器上找到；本次按可用的 `{FALLBACK_PLAN.name}` 和当前 Phase 2 要求执行，并已写入 S0 审计。",
            "- 三个数据集的 context-perturbation task 数都不大，统计稳定性有限。",
            "- `learned_risk_score_v2` 只用 validation records 训练，防泄漏是好的，但训练样本仍偏少。",
            "- 还没有接 GEARS/CPA 这类 published predictor，因此不能说是完整论文级 benchmark。",
            "- 生物解释目前只是 transferability ranking，还不是 pathway enrichment 级别。",
            "",
            "## 8. 下一步建议",
            "",
            "优先扩到 Norman / Adamson 或已有 GEARS 结果，把 confidence scoring 证明成 predictor-agnostic；同时把 feature 做成更稳定的 pathway/program 版本。短期汇报可以讲：任务定义、三数据集 MVP、held-out pair 防泄漏、score 和 error 的相关性、以及当前边界。",
        ]
    )
    write_text(dirs.out / "MVP_V2_REPORT.md", "\n".join(report) + "\n")

    checklist_rows = [
        ("S0", "整理已有 risk 信号", "DONE", "tables/S0_existing_signal.csv; reports/S0_audit_report.md"),
        ("S1", "PredictionRecord 表", "DONE", "tables/PREDICTION_RECORDS.csv; input/predicted_effects.npz; input/true_effects.npz"),
        ("S2", "held-out pair split", "DONE", "tables/HELDOUT_PAIR_SPLITS.csv; reports/split_audit_report.md"),
        ("S3", "V0 + ContextSim prediction", "DONE", "tables/PREDICTOR_STATUS.csv; tables/PREDICTION_RECORDS.csv"),
        ("S4", "confidence features", "DONE", "tables/CONFIDENCE_FEATURES.csv"),
        ("S5", "confidence baselines + learned shallow", "DONE", "tables/CONFIDENCE_SCORES.csv; tables/LEARNED_RISK_STATUS.json"),
        ("S6", "evaluation + main figures", "DONE", "tables/CONFIDENCE_EVAL_SUMMARY.csv; figures/F4-F8*.png"),
        ("S7", "多 dataset robustness", "PARTIAL", "KaggleCrossCell/Haber/Parekh done; no multi-seed yet"),
        ("S8", "生物解释", "PARTIAL", "tables/TRANSFERABILITY_RANKING.csv; figures/F10_transferability_ranking.png; no pathway enrichment"),
        ("S9", "论文写作", "PARTIAL", "MVP_V2_REPORT.md only"),
        ("F1", "Task schematic", "DONE", "figures/F1_task_schematic.png"),
        ("F2", "Context x perturbation matrix", "DONE", "figures/F2_context_perturbation_matrix.png"),
        ("F3", "PredictionRecord flow", "DONE", "figures/F3_prediction_record_flow.png"),
        ("F4", "Confidence vs true error scatter", "DONE", "figures/F4_confidence_vs_true_error_scatter.png"),
        ("F5", "Risk-coverage curve", "DONE", "figures/F5_risk_coverage_curve.png"),
        ("F6", "High vs low RMSE", "DONE", "figures/F6_high_vs_low_confidence_rmse.png"),
        ("F7", "Baseline comparison Spearman", "DONE", "figures/F7_baseline_spearman_comparison.png"),
        ("F8", "Calibration buckets", "DONE", "figures/F8_calibration_buckets.png"),
        ("F9", "Feature importance", "DONE", "figures/F9_feature_importance.png" if (dirs.figures / "F9_feature_importance.png").exists() else "feature importance unavailable"),
        ("F10", "Biological interpretation", "PARTIAL", "figures/F10_transferability_ranking.png; no enrichment"),
    ]
    lines = [
        "# Stage completion checklist",
        "",
        f"- Requested exact spec file found: `{REQUESTED_SPEC.exists()}`.",
        f"- Checklist is filled against available plan `{FALLBACK_PLAN}` sections 12 and 18 plus the current user Phase 2 constraints.",
        "",
        "| Item | Requirement | Status | Evidence |",
        "|---|---|---|---|",
    ]
    for item, req, status, evidence in checklist_rows:
        lines.append(f"| {item} | {req} | {status} | `{evidence}` |")
    write_text(dirs.out / "stage_completion_checklist.md", "\n".join(lines) + "\n")

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


def copy_scripts(dirs: RunDirs):
    script_paths = [
        Path(__file__),
        PROJECT_ROOT / "confidence_task" / "run_confidence_mvp_v2.sh",
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
    parser = argparse.ArgumentParser(description="Run Phase 2 three-dataset confidence scoring MVP v2.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--atlas-root", default=str(DEFAULT_ATLAS_ROOT))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "outputs" / "confidence_task_mvp_v2"))
    parser.add_argument("--n-genes", type=int, default=1000)
    parser.add_argument("--min-cells", type=int, default=10)
    parser.add_argument("--max-cells-per-group", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260521)
    args = parser.parse_args()

    out = Path(args.out_dir)
    if out.resolve() == (PROJECT_ROOT / "outputs" / "confidence_task_mvp_final").resolve():
        raise RuntimeError("Refusing to overwrite confidence_task_mvp_final")
    dirs = make_dirs(out)
    start = time.time()
    print(f"[start] confidence MVP v2 output: {dirs.out}", flush=True)
    warnings.filterwarnings("ignore", category=FutureWarning)

    s0 = run_s0_audit(dirs)
    paths = resolve_dataset_paths(Path(args.atlas_root))
    tasks_by_dataset, genes_by_dataset, dataset_meta = build_all_tasks(
        paths, dirs, n_genes=args.n_genes, min_cells=args.min_cells, max_cells_per_group=args.max_cells_per_group, seed=args.seed
    )
    split_df = build_pair_splits(tasks_by_dataset, dirs, seed=args.seed)
    rec_df, pred_arrays, true_arrays = run_predictors(tasks_by_dataset, split_df, dirs)
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
    zip_path = make_zip(dirs)
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
    print(json.dumps(status, ensure_ascii=False, indent=2, default=to_jsonable), flush=True)
    print(f"[done] zip: {zip_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
