#!/usr/bin/env python3
"""Select a quality-gated train-source predictor using Jiang24 validation only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ALPHAS = (0.05, 0.10, 0.25, 0.50, 1.00)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    writer(temp)
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-cache", "tasks", "target-controls", "target-control-status",
                 "validation-control-cache", "validation-truth", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("E245 validation result exists; refusing overwrite")
    source_status = json.loads((args.source_cache / "E245_VALIDATION_SOURCE_CACHE_STATUS.json").read_text())
    if (source_status.get("status") != "PASS"
            or source_status.get("test_perturbed_expression_rows_read") != 0
            or source_status.get("validation_perturbed_expression_rows_read") != 0
            or source_status.get("min_source_cells") != 30):
        raise RuntimeError("E245 train-only source cache failed")
    paths = {"effects": args.source_cache / "E245_VALIDATION_SOURCE_EFFECTS.npy",
             "manifest": args.source_cache / "E245_VALIDATION_SOURCE_MANIFEST.csv",
             "gene_axis": args.source_cache / "E245_VALIDATION_GENE_AXIS.csv"}
    for name, path in paths.items():
        if sha256(path) != source_status[name + "_sha256"]:
            raise RuntimeError(f"changed source-cache input: {name}")
    if sha256(args.tasks) != source_status["validation_task_manifest_sha256"]:
        raise RuntimeError("validation task order changed")
    control_status = json.loads(args.target_control_status.read_text())
    if (control_status.get("status") != "PASS"
            or control_status.get("control_centroids_sha256") != sha256(args.target_controls)
            or control_status.get("test_perturbed_expression_rows_read") != 0):
        raise RuntimeError("target validation controls changed")
    tasks = pd.read_csv(args.tasks)
    control_tasks = pd.read_csv(args.target_controls.parent / "E208_VALIDATION_TASKS.csv")
    if not tasks.equals(control_tasks):
        raise RuntimeError("validation controls use a different task order")
    with np.load(args.validation_control_cache, allow_pickle=False) as packed:
        control_gene_axis = packed["gene_names"].astype(str)
    source_gene_axis = pd.read_csv(paths["gene_axis"]).gene_name.astype(str).to_numpy()
    if not np.array_equal(control_gene_axis, source_gene_axis):
        raise RuntimeError("train-source and validation-control gene axes differ")
    controls = np.load(args.target_controls, allow_pickle=False).astype(np.float64)
    truth = np.load(args.validation_truth, allow_pickle=False).astype(np.float64)
    effects = np.load(paths["effects"], allow_pickle=False).astype(np.float64)
    source = pd.read_csv(paths["manifest"])
    if (controls.shape != truth.shape or controls.shape != (216, 15473)
            or effects.shape != (len(source), 15473)
            or not np.isfinite(controls).all() or not np.isfinite(truth).all()):
        raise RuntimeError("E245 validation arrays do not align")
    rows = []
    for index, task in enumerate(tasks.itertuples(index=False)):
        history = source.loc[source.condition.eq(str(task.condition))
                             & source.source_context_id.ne(f"{task.cell_type}|{task.treatment}")]
        rows.append({"index": index, "task_id": task.task_id,
                     "cell_type": task.cell_type, "treatment": task.treatment,
                     "condition": task.condition, "n_sources": len(history),
                     "eligible": len(history) >= 2,
                     "source_rows": history.effect_row.to_numpy(np.int64)})
    if sum(item["eligible"] for item in rows) != 212:
        raise RuntimeError("E245 validation eligibility changed")
    eligible = [item for item in rows if item["eligible"]]
    indexes = np.asarray([item["index"] for item in eligible], dtype=np.int64)
    source_mean = np.stack([effects[item["source_rows"]].mean(axis=0) for item in eligible])
    y = truth[indexes] - controls[indexes]
    no_change = np.mean(y**2, axis=1)
    groups = pd.DataFrame(eligible).groupby(["cell_type", "treatment"], sort=True).indices
    if len(groups) != 12:
        raise RuntimeError("E245 lost a validation state")
    candidates = []
    tasks_out = pd.DataFrame([{key: value for key, value in item.items() if key != "source_rows"}
                              for item in eligible])
    tasks_out["no_change_mse"] = no_change
    contexts_out = []
    for alpha in ALPHAS:
        model_error = np.mean((alpha * source_mean - y) ** 2, axis=1)
        mean_model = float(model_error.mean())
        mean_baseline = float(no_change.mean())
        contexts_not_worse = 0
        for (cell, treatment), positions in groups.items():
            model_context = float(model_error[positions].mean())
            base_context = float(no_change[positions].mean())
            contexts_not_worse += model_context <= base_context
            contexts_out.append({"alpha": alpha, "cell_type": cell, "treatment": treatment,
                                 "n_tasks": len(positions), "model_mse": model_context,
                                 "no_change_mse": base_context,
                                 "not_worse": model_context <= base_context})
        relative_gain = float(1 - mean_model / mean_baseline)
        passed = relative_gain >= 0.02 and contexts_not_worse >= 8
        candidates.append({"alpha": alpha, "n_tasks": len(eligible), "n_states": len(groups),
                           "model_mean_mse": mean_model, "no_change_mean_mse": mean_baseline,
                           "relative_gain": relative_gain,
                           "states_not_worse": contexts_not_worse, "competence_gate": "PASS" if passed else "FAIL"})
        tasks_out[f"mse_alpha_{alpha:g}"] = model_error
    candidates_frame = pd.DataFrame(candidates)
    passing = candidates_frame.loc[candidates_frame.competence_gate.eq("PASS")]
    selected = (float(passing.sort_values(["model_mean_mse", "alpha"]).iloc[0].alpha)
                if len(passing) else None)
    decision = "VALIDATION_COMPETENCE_PASS" if selected is not None else "BLOCKED_UPSTREAM_COMPETENCE"
    output.mkdir(parents=True, exist_ok=True)
    table_path = output / "E245_VALIDATION_ALPHA_TABLE.csv"
    context_path = output / "E245_VALIDATION_CONTEXTS.csv"
    task_path = output / "E245_VALIDATION_TASKS.csv"
    atomic(table_path, lambda p: candidates_frame.to_csv(p, index=False))
    atomic(context_path, lambda p: pd.DataFrame(contexts_out).to_csv(p, index=False))
    atomic(task_path, lambda p: tasks_out.to_csv(p, index=False))
    record = {"experiment": "E245_jiang24_quality_gated_source_transfer",
              "status": decision, "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
              "n_validation_tasks": 216, "n_eligible": 212, "n_abstain": 4,
              "alphas": list(ALPHAS), "selected_alpha": selected,
              "gate": "mean_MSE_gain>=0.02_and_8_of_12_states_not_worse",
              "source_cache_status_sha256": sha256(args.source_cache / "E245_VALIDATION_SOURCE_CACHE_STATUS.json"),
              "validation_truth_sha256": sha256(args.validation_truth),
              "test_perturbed_expression_rows_read": 0,
              "table_sha256": sha256(table_path), "contexts_sha256": sha256(context_path),
              "tasks_sha256": sha256(task_path)}
    atomic(output / "E245_VALIDATION_STATUS.json",
           lambda p: p.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n"))
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
