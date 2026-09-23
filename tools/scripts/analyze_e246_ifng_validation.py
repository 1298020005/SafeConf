#!/usr/bin/env python3
"""Select E246's IFNG history weight using validation errors only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from run_e235_formal_evaluation import state_metrics


ALPHA = 0.10
WEIGHTS = (0.0, 0.2, 0.4, 0.6, 1.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    writer(tmp)
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-cache", "validation-tasks", "target-controls", "validation-truth", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("E246 validation output exists; refusing overwrite")
    cache = json.loads((args.source_cache / "E245_VALIDATION_SOURCE_CACHE_STATUS.json").read_text())
    if (cache.get("status") != "PASS" or cache.get("test_perturbed_expression_rows_read") != 0
            or cache.get("validation_task_manifest_sha256") != sha256(args.validation_tasks)):
        raise RuntimeError("E245 train-only source cache failed")
    effect_path = args.source_cache / "E245_VALIDATION_SOURCE_EFFECTS.npy"
    manifest_path = args.source_cache / "E245_VALIDATION_SOURCE_MANIFEST.csv"
    if sha256(effect_path) != cache["effects_sha256"] or sha256(manifest_path) != cache["manifest_sha256"]:
        raise RuntimeError("E245 source effects changed")
    effects = np.load(effect_path, allow_pickle=False).astype(np.float64)
    source = pd.read_csv(manifest_path)
    tasks = pd.read_csv(args.validation_tasks)
    controls = np.load(args.target_controls, allow_pickle=False).astype(np.float64)
    truth = np.load(args.validation_truth, allow_pickle=False).astype(np.float64)
    if controls.shape != truth.shape or controls.shape != (216, 15473):
        raise RuntimeError("E246 validation arrays changed")
    rows = []
    for position, task in enumerate(tasks.itertuples(index=False)):
        if task.treatment != "IFNG":
            continue
        history = source.loc[source.condition.eq(str(task.condition))
                             & source.source_context_id.ne(f"{task.cell_type}|{task.treatment}")]
        if len(history) < 2:
            continue
        vectors = effects[history.effect_row.to_numpy(np.int64)]
        mean = vectors.mean(axis=0)
        delta = ALPHA * mean
        rows.append({"task_id": task.task_id, "cell_type": task.cell_type,
                     "treatment": task.treatment, "condition": task.condition,
                     "n_sources": len(history),
                     "M": float(np.sqrt(np.mean(delta**2))),
                     "H": float(np.sqrt(np.mean((vectors - mean[None, :]) ** 2))),
                     "full_gene_rmse": float(np.sqrt(np.mean((controls[position] + delta - truth[position]) ** 2))),
                     "no_change_rmse": float(np.sqrt(np.mean((controls[position] - truth[position]) ** 2))),
                     "test_perturbed_expression_rows_read": 0})
    frame = pd.DataFrame(rows)
    if (len(frame) != 80 or frame.condition.nunique() != 20
            or frame.groupby("cell_type").size().to_dict() != {"hap1": 20, "ht29": 20, "k562": 20, "mcf7": 20}):
        raise RuntimeError("IFNG validation task eligibility changed")
    for _, positions in frame.groupby("cell_type", sort=True).groups.items():
        positions = list(positions)
        n = len(positions)
        for feature in ("M", "H"):
            frame.loc[positions, "rank_" + feature] = frame.loc[positions, feature].rank(method="average") / n
    for weight in WEIGHTS:
        key = f"risk_w_{str(weight).replace('.', '_')}"
        frame[key] = (1 - weight) * frame.rank_M + weight * frame.rank_H
    summary = []
    for weight in WEIGHTS:
        key = f"risk_w_{str(weight).replace('.', '_')}"
        per_state = [state_metrics(block, key, 0.2)["utility"]
                     for _, block in frame.groupby("cell_type", sort=True)]
        summary.append({"weight_H": weight, "macro_utility_20": float(np.mean(per_state)),
                        "hap1": per_state[0], "ht29": per_state[1],
                        "k562": per_state[2], "mcf7": per_state[3]})
    summary = pd.DataFrame(summary)
    baseline = summary.loc[summary.weight_H.eq(0)].iloc[0]
    states = ("hap1", "ht29", "k562", "mcf7")
    summary["delta_vs_M"] = summary.macro_utility_20 - float(baseline.macro_utility_20)
    summary["positive_states"] = [sum(float(row[state]) > float(baseline[state]) for state in states)
                                   for _, row in summary.iterrows()]
    summary["development_gate"] = ((summary.weight_H.gt(0) & summary.delta_vs_M.ge(0.03)
                                     & summary.positive_states.ge(3)).map({True: "PASS", False: "FAIL"}))
    passing = summary.loc[summary.development_gate.eq("PASS")]
    selected = (float(passing.sort_values(["macro_utility_20", "weight_H"],
                                          ascending=[False, True]).iloc[0].weight_H)
                if len(passing) else None)
    output.mkdir(parents=True, exist_ok=True)
    task_path = output / "E246_VALIDATION_TASKS.csv"
    table_path = output / "E246_WEIGHT_GRID.csv"
    atomic(task_path, lambda p: frame.to_csv(p, index=False))
    atomic(table_path, lambda p: summary.to_csv(p, index=False))
    record = {"experiment": "E246_ifng_history_routing", "stage": "VALIDATION_WEIGHT_SELECTION",
              "status": "DEVELOPMENT_CANDIDATE_READY" if selected is not None else "NO_RISK_CANDIDATE",
              "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
              "alpha": ALPHA, "candidate_weights_H": list(WEIGHTS), "selected_weight_H": selected,
              "n_validation_tasks": len(frame), "n_genes": frame.condition.nunique(),
              "n_states": frame.cell_type.nunique(),
              "test_perturbed_expression_rows_read": 0,
              "task_table_sha256": sha256(task_path), "weight_grid_sha256": sha256(table_path)}
    atomic(output / "E246_DEVELOPMENT_STATUS.json",
           lambda p: p.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n"))
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
