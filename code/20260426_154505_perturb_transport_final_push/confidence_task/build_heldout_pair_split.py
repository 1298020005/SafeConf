#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "03_code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from build_context_splits import build_effect_tasks, read_scan_table


def find_dataset_path(atlas_root: Path, dataset_name: str) -> Path:
    scan = read_scan_table(atlas_root)
    hits = scan[scan["study_family"].astype(str) == dataset_name]
    hits = hits[hits["local_path"].map(lambda x: Path(str(x)).exists())]
    if hits.empty:
        raise FileNotFoundError(f"No existing h5ad found for study_family={dataset_name!r}")
    return Path(str(hits.iloc[0]["local_path"]))


def task_table(tasks: list[dict], dataset_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "task_id": list(range(len(tasks))),
            "dataset_name": dataset_name,
            "context": [str(t["context"]) for t in tasks],
            "perturbation": [str(t["perturbation"]) for t in tasks],
            "n_cells": [int(t.get("n_cells", 0)) for t in tasks],
        }
    )


def eligible_task_ids(tasks: list[dict]) -> list[int]:
    context_counts = Counter(str(t["context"]) for t in tasks)
    perturbation_counts = Counter(str(t["perturbation"]) for t in tasks)
    ids = []
    for i, task in enumerate(tasks):
        if context_counts[str(task["context"])] >= 2 and perturbation_counts[str(task["perturbation"])] >= 2:
            ids.append(i)
    return ids


def assign_folds(tasks: list[dict], candidate_ids: list[int], n_folds: int, seed: int) -> dict[int, int]:
    rng = np.random.default_rng(seed)
    context_total = Counter(str(t["context"]) for t in tasks)
    perturbation_total = Counter(str(t["perturbation"]) for t in tasks)
    order = list(candidate_ids)
    rng.shuffle(order)
    order.sort(
        key=lambda i: (
            perturbation_total[str(tasks[i]["perturbation"])],
            context_total[str(tasks[i]["context"])],
            rng.random(),
        )
    )

    fold_sizes = Counter()
    fold_context_counts: dict[int, Counter] = defaultdict(Counter)
    fold_pert_counts: dict[int, Counter] = defaultdict(Counter)
    assignment: dict[int, int] = {}

    for task_id in order:
        ctx = str(tasks[task_id]["context"])
        pert = str(tasks[task_id]["perturbation"])
        candidate_folds = sorted(range(n_folds), key=lambda f: (fold_sizes[f], rng.random()))
        chosen = None
        for fold_id in candidate_folds:
            if fold_context_counts[fold_id][ctx] + 1 >= context_total[ctx]:
                continue
            if fold_pert_counts[fold_id][pert] + 1 >= perturbation_total[pert]:
                continue
            chosen = fold_id
            break
        if chosen is None:
            chosen = candidate_folds[0]
        assignment[task_id] = chosen
        fold_sizes[chosen] += 1
        fold_context_counts[chosen][ctx] += 1
        fold_pert_counts[chosen][pert] += 1
    return assignment


def choose_val_ids(
    tasks: list[dict],
    train_pool: list[int],
    test_ids: set[int],
    seed: int,
    max_fraction: float,
) -> set[int]:
    rng = np.random.default_rng(seed)
    target = max(1, int(round(len(train_pool) * max_fraction)))
    target = min(target, max(1, len(train_pool) // 5))
    shuffled = list(train_pool)
    rng.shuffle(shuffled)
    val_ids: set[int] = set()

    def supports_hold(candidate_val: set[int]) -> bool:
        train_after = set(train_pool) - candidate_val
        train_contexts = {str(tasks[i]["context"]) for i in train_after}
        train_perts = {str(tasks[i]["perturbation"]) for i in train_after}
        train_pairs = {(str(tasks[i]["context"]), str(tasks[i]["perturbation"])) for i in train_after}
        for tid in test_ids:
            pair = (str(tasks[tid]["context"]), str(tasks[tid]["perturbation"]))
            if pair in train_pairs:
                return False
            if pair[0] not in train_contexts or pair[1] not in train_perts:
                return False
        return True

    for task_id in shuffled:
        if len(val_ids) >= target:
            break
        proposed = set(val_ids)
        proposed.add(task_id)
        if supports_hold(proposed):
            val_ids = proposed
    return val_ids


def build_split_rows(
    tasks: list[dict],
    dataset_name: str,
    n_folds: int,
    seed: int,
    val_fraction: float,
) -> tuple[pd.DataFrame, dict]:
    table = task_table(tasks, dataset_name)
    candidates = eligible_task_ids(tasks)
    assignment = assign_folds(tasks, candidates, n_folds, seed)

    rows = []
    fold_summaries = []
    all_ids = set(range(len(tasks)))
    for fold_id in range(n_folds):
        test_ids = {task_id for task_id, assigned in assignment.items() if assigned == fold_id}
        if not test_ids:
            raise RuntimeError(f"Fold {fold_id} has no test tasks; reduce fold count.")
        train_pool = sorted(all_ids - test_ids)
        val_ids = choose_val_ids(tasks, train_pool, test_ids, seed + 1000 + fold_id, val_fraction)
        train_ids = set(train_pool) - val_ids
        train_pairs = {(str(tasks[i]["context"]), str(tasks[i]["perturbation"])) for i in train_ids}
        train_contexts = {str(tasks[i]["context"]) for i in train_ids}
        train_perts = {str(tasks[i]["perturbation"]) for i in train_ids}

        leakage = []
        missing_context = []
        missing_perturbation = []
        for task_id in sorted(test_ids):
            ctx = str(tasks[task_id]["context"])
            pert = str(tasks[task_id]["perturbation"])
            if (ctx, pert) in train_pairs:
                leakage.append(task_id)
            if ctx not in train_contexts:
                missing_context.append(task_id)
            if pert not in train_perts:
                missing_perturbation.append(task_id)
        if leakage or missing_context or missing_perturbation:
            raise RuntimeError(
                f"Fold {fold_id} failed leakage/support checks: "
                f"leakage={leakage}, missing_context={missing_context}, "
                f"missing_perturbation={missing_perturbation}"
            )

        for task_id in sorted(all_ids):
            ctx = str(tasks[task_id]["context"])
            pert = str(tasks[task_id]["perturbation"])
            if task_id in test_ids:
                split = "test"
            elif task_id in val_ids:
                split = "val"
            else:
                split = "train"
            rows.append(
                {
                    "task_id": int(task_id),
                    "dataset_name": dataset_name,
                    "context": ctx,
                    "perturbation": pert,
                    "fold_id": int(fold_id),
                    "split": split,
                    "pair_seen_in_train": bool((ctx, pert) in train_pairs),
                    "perturbation_seen_in_train": bool(pert in train_perts),
                    "context_seen_in_train": bool(ctx in train_contexts),
                }
            )
        fold_summaries.append(
            {
                "fold_id": int(fold_id),
                "n_train": int(len(train_ids)),
                "n_val": int(len(val_ids)),
                "n_test": int(len(test_ids)),
                "n_train_contexts": int(len(train_contexts)),
                "n_train_perturbations": int(len(train_perts)),
                "test_pair_leakage_count": int(len(leakage)),
                "test_context_missing_count": int(len(missing_context)),
                "test_perturbation_missing_count": int(len(missing_perturbation)),
            }
        )
    return pd.DataFrame(rows), {"folds": fold_summaries, "n_candidate_test_tasks": len(candidates)}


def write_report(path: Path, split_df: pd.DataFrame, summary: dict) -> None:
    test = split_df[split_df["split"] == "test"]
    lines = [
        "# Held-out context-perturbation pair split report",
        "",
        f"- Dataset: `{summary['dataset_name']}`",
        f"- Dataset path: `{summary['dataset_path']}`",
        f"- n_tasks: {summary['n_tasks']}",
        f"- n_contexts: {summary['n_contexts']}",
        f"- n_perturbations: {summary['n_perturbations']}",
        f"- requested_folds: {summary['requested_folds']}",
        f"- actual_folds: {summary['actual_folds']}",
        f"- fold_downshift_reason: {summary.get('fold_downshift_reason') or 'none'}",
        "",
        "## Leakage checks",
        "",
        f"- test pair leakage count: {int(test['pair_seen_in_train'].sum())}",
        f"- test perturbation missing in train count: {int((~test['perturbation_seen_in_train']).sum())}",
        f"- test context missing in train count: {int((~test['context_seen_in_train']).sum())}",
        "",
        "## Fold summary",
        "",
        "| fold_id | n_train | n_val | n_test | train_contexts | train_perturbations | leakage | missing_context | missing_perturbation |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary["folds"]:
        lines.append(
            f"| {row['fold_id']} | {row['n_train']} | {row['n_val']} | {row['n_test']} | "
            f"{row['n_train_contexts']} | {row['n_train_perturbations']} | "
            f"{row['test_pair_leakage_count']} | {row['test_context_missing_count']} | "
            f"{row['test_perturbation_missing_count']} |"
        )
    lines.extend(
        [
            "",
            "## Suitability",
            "",
            "This split is suitable for confidence scoring MVP if leakage is zero and every test task has both its context and perturbation represented in train.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build held-out context-perturbation pair splits for the confidence MVP.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--atlas-root", default="/home/yyf/datasets/singlecell_perturbation_atlas")
    parser.add_argument("--dataset-name", default="KaggleCrossCell")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "outputs" / "confidence_task_mvp"))
    parser.add_argument("--n-genes", type=int, default=1500)
    parser.add_argument("--min-cells", type=int, default=15)
    parser.add_argument("--max-cells-per-group", type=int, default=400)
    parser.add_argument("--task-seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=20260521)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-test-per-fold", type=int, default=3)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    split_dir = out_dir / "splits"
    report_dir = out_dir / "reports"
    split_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(args.dataset_path) if args.dataset_path else find_dataset_path(Path(args.atlas_root), args.dataset_name)
    tasks, genes, meta = build_effect_tasks(
        dataset_path,
        args.dataset_name,
        n_genes=args.n_genes,
        min_cells=args.min_cells,
        max_cells_per_group=args.max_cells_per_group,
        seed=args.task_seed,
    )
    if not tasks:
        raise RuntimeError(f"No tasks built for {args.dataset_name}: {meta}")

    candidates = eligible_task_ids(tasks)
    requested_folds = int(args.folds)
    actual_folds = requested_folds
    downshift_reason = ""
    if len(candidates) < requested_folds * args.min_test_per_fold:
        actual_folds = 3
        downshift_reason = (
            f"candidate tasks ({len(candidates)}) < requested_folds * min_test_per_fold "
            f"({requested_folds * args.min_test_per_fold}); downshifted to 3 folds"
        )
    split_df, fold_info = build_split_rows(tasks, args.dataset_name, actual_folds, args.split_seed, args.val_fraction)
    split_path = split_dir / "kagglecrosscell_heldout_pair_split.csv"
    summary_path = split_dir / "kagglecrosscell_heldout_pair_split_summary.json"
    report_path = report_dir / "heldout_pair_split_report.md"
    split_df.to_csv(split_path, index=False)

    summary = {
        "dataset_name": args.dataset_name,
        "dataset_path": str(dataset_path),
        "n_tasks": int(len(tasks)),
        "n_contexts": int(meta.get("n_contexts", 0)),
        "n_perturbations": int(meta.get("n_perturbations", 0)),
        "context_col": meta.get("context_col"),
        "perturbation_col": meta.get("perturbation_col"),
        "n_genes": int(len(genes)),
        "n_candidate_test_tasks": int(fold_info["n_candidate_test_tasks"]),
        "requested_folds": int(requested_folds),
        "actual_folds": int(actual_folds),
        "fold_downshift_reason": downshift_reason,
        "task_seed": int(args.task_seed),
        "split_seed": int(args.split_seed),
        "min_cells": int(args.min_cells),
        "max_cells_per_group": int(args.max_cells_per_group),
        "folds": fold_info["folds"],
        "outputs": {
            "split_csv": str(split_path),
            "summary_json": str(summary_path),
            "report_md": str(report_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(report_path, split_df, summary)
    print(json.dumps({"split_csv": str(split_path), "summary_json": str(summary_path), "report_md": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
