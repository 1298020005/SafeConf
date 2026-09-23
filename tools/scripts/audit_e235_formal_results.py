#!/usr/bin/env python3
"""Independently recompute E235's registered utility, bootstrap, and pass gates.

This audits metrics from the committed task-level errors; it does not reread the
93 GB expression source and must not be described as an independent truth audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def utility(block: pd.DataFrame, method: str, occurrences: np.ndarray | None = None) -> float:
    n = len(block)
    errors = block.full_gene_rmse.to_numpy(np.float64)
    risks = block[method].to_numpy(np.float64)
    if occurrences is None:
        occurrences = np.zeros(n, dtype=np.int64)
    tie_keys = np.asarray([
        int(hashlib.sha256(f"E235_TIE_V1\0{task}\0{int(occ)}".encode()).hexdigest()[:16], 16)
        for task, occ in zip(block.task_id.astype(str), occurrences, strict=True)
    ], dtype=np.uint64)
    k = math.ceil(0.2 * n)
    chosen = np.lexsort((tie_keys, -risks))[:k]
    best = np.lexsort((tie_keys, -errors))[:k]
    random_mean = errors.mean()
    ideal_gain = errors[best].mean() - random_mean
    return float((errors[chosen].mean() - random_mean) / ideal_gain) if ideal_gain > 1e-15 else float("nan")


def macro(frame: pd.DataFrame, method: str, occurrences: np.ndarray | None = None) -> float:
    values = []
    for _, positions in frame.groupby(["cell_type", "treatment"], sort=True).indices.items():
        block = frame.iloc[positions]
        values.append(utility(block, method, None if occurrences is None else occurrences[positions]))
    if len(values) != 12:
        raise RuntimeError("audit did not retain 12 states")
    return float(np.mean(values))


def near(a: float, b: float) -> bool:
    return bool(np.isclose(a, b, rtol=1e-9, atol=1e-11, equal_nan=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = args.formal_dir
    status = json.loads((directory / "E235_FORMAL_EVALUATION_STATUS.json").read_text())
    if status.get("status") != "COMPLETE":
        raise RuntimeError("formal evaluation incomplete")
    for name, expected in status["output_sha256"].items():
        if sha256(directory / name) != expected:
            raise RuntimeError(f"changed formal result: {name}")
    tasks = pd.read_csv(directory / "E235_FORMAL_TASK_RESULTS.csv")
    summary = pd.read_csv(directory / "E235_FORMAL_SUMMARY.csv")
    states = pd.read_csv(directory / "E235_FORMAL_STATE_RESULTS.csv")
    bootstrap = pd.read_csv(directory / "E235_GENE_CLUSTER_BOOTSTRAP.csv.gz")
    if (len(tasks) != 224 or tasks.task_id.nunique() != 224 or tasks.condition.nunique() != 53
            or tasks.n_truth_cells.sum() != 214_901 or len(bootstrap) != 2000):
        raise RuntimeError("formal task/count/draw contract changed")
    if not np.isfinite(tasks[["rank_M", "score_M_plus_H", "full_gene_rmse"]].to_numpy(float)).all():
        raise RuntimeError("nonfinite score or error")
    observed = {method: macro(tasks, method) for method in ("rank_M", "score_M_plus_H")}
    reported = summary.loc[summary.budget.eq(0.2)].set_index("method")
    for method, point in observed.items():
        if not near(point, float(reported.loc[method, "utility"])):
            raise RuntimeError(f"macro utility audit failed for {method}")
    paired = {}
    for (cell, treatment), block in tasks.groupby(["cell_type", "treatment"], sort=True):
        for method in observed:
            point = utility(block, method)
            published = states.loc[states.cell_type.eq(cell) & states.treatment.eq(treatment)
                                   & states.method.eq(method) & states.budget.eq(0.2)]
            if len(published) != 1 or not near(point, float(published.iloc[0].utility)):
                raise RuntimeError(f"state utility audit failed: {cell}/{treatment}/{method}")
            paired[(cell, treatment, method)] = point
    positive = sum(paired[cell, treatment, "score_M_plus_H"] > paired[cell, treatment, "rank_M"]
                   for cell, treatment in tasks[["cell_type", "treatment"]].drop_duplicates().itertuples(index=False))
    if positive != status["positive_utility_states"]:
        raise RuntimeError("positive-state count audit failed")

    groups = sorted(tasks.condition.astype(str).unique())
    genes = tasks.condition.astype(str).to_numpy()
    by_gene = [np.flatnonzero(genes == gene) for gene in groups]
    rng = np.random.default_rng(20260923)
    for draw in range(2000):
        chosen = rng.integers(0, len(groups), len(groups))
        positions = np.concatenate([by_gene[int(index)] for index in chosen])
        occurrences = np.concatenate([
            np.full(len(by_gene[int(index)]), repeat, dtype=np.int64)
            for repeat, index in enumerate(chosen)
        ])
        sample = tasks.iloc[positions].reset_index(drop=True)
        delta = macro(sample, "score_M_plus_H", occurrences) - macro(sample, "rank_M", occurrences)
        if not near(delta, float(bootstrap.iloc[draw].delta_utility_20)):
            raise RuntimeError(f"bootstrap audit failed at draw {draw}")
    valid = bootstrap.delta_utility_20.dropna()
    ci = (float(valid.quantile(0.025)), float(valid.quantile(0.975)))
    reported_ci = status["gene_cluster_bootstrap_ci_95"]["delta_utility_20"]
    if not all(near(a, b) for a, b in zip(ci, reported_ci, strict=True)):
        raise RuntimeError("gene-cluster CI audit failed")
    delta = observed["score_M_plus_H"] - observed["rank_M"]
    gates = {"macro_utility_delta_at_least_0_02": delta >= 0.02,
             "gene_cluster_ci_lower_gt_zero": ci[0] > 0,
             "at_least_9_of_12_states_positive": positive >= 9}
    if (gates != status["gates"] or not near(delta, float(status["primary_delta_utility_20"]))
            or status["decision"] != ("SUPPORTED_IN_FIXED_SCOPE" if all(gates.values()) else "NOT_SUPPORTED")):
        raise RuntimeError("formal decision/gate audit failed")
    result = {"experiment": "E235_jiang24_fixed_history_dispersion",
              "status": "PASS", "audit_scope": "metrics_and_gene_bootstrap_only_no_H5_truth_reread",
              "audited_at": datetime.now().astimezone().isoformat(timespec="seconds"),
              "primary_delta_utility_20": delta, "ci_95": ci, "positive_states": positive,
              "decision": status["decision"], "n_bootstrap_recomputed": len(bootstrap)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
