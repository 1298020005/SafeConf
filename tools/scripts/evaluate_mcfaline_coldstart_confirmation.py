#!/usr/bin/env python3
"""One-shot evaluation of sealed McFaline cold-start risk predictions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr


SEED = 20260930


def categorical(group: h5py.Group, name: str) -> np.ndarray:
    node = group[name]
    if isinstance(node, h5py.Group):
        categories = np.asarray(node["categories"]).astype(str)
        codes = np.asarray(node["codes"], dtype=int)
        values = np.empty(len(codes), dtype=object)
        values[codes < 0] = None
        valid = codes >= 0
        values[valid] = categories[codes[valid]]
        return values.astype(str)
    return np.asarray(node).astype(str)


def accumulate(
    sums: np.ndarray, counts: np.ndarray, codes: np.ndarray, values: np.ndarray
) -> None:
    valid = codes >= 0
    if not valid.any():
        return
    use_codes, use_values = codes[valid], values[valid]
    order = np.argsort(use_codes, kind="stable")
    use_codes, use_values = use_codes[order], use_values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(use_codes)) + 1]
    unique = use_codes[starts]
    sums[unique] += np.add.reduceat(use_values, starts, axis=0)
    counts[unique] += np.diff(np.r_[starts, len(use_codes)])


def aggregate_test_truth(
    h5ad: Path,
    split_path: Path,
    tasks: pd.DataFrame,
    gene_ids: list[str],
    controls: np.ndarray,
    chunk_rows: int,
) -> np.ndarray:
    split = pd.read_csv(split_path, header=None, names=["cell_id", "split"])
    split_map = pd.Series(split["split"].to_numpy(str), index=split.cell_id.astype(str)).to_dict()
    task_index = pd.Series(np.arange(len(tasks)), index=tasks.task_id.astype(str)).to_dict()
    with h5py.File(h5ad, "r") as handle:
        obs = handle["obs"]
        cell_ids = categorical(obs, "_index")
        roles = np.asarray([split_map.get(value, "missing") for value in cell_ids], dtype=object)
        if (roles == "missing").any():
            raise RuntimeError("split failed to align")
        control = categorical(obs, "control") == "1"
        condition = categorical(obs, "condition")
        context = categorical(obs, "cell_type")
        treatment = categorical(obs, "treatment")
        ids = np.asarray(
            [f"{a}::{b}::{c}" for a, b, c in zip(condition, context, treatment)],
            dtype=object,
        )
        codes = np.full(len(ids), -1, dtype=int)
        eligible = (roles == "test") & ~control
        for row in np.flatnonzero(eligible):
            codes[row] = task_index.get(str(ids[row]), -1)
        if (codes[eligible] < 0).any():
            missing = sorted(set(ids[eligible][codes[eligible] < 0]))[:10]
            raise RuntimeError(f"test treated tasks missing from sealed manifest: {missing}")
        all_genes = np.asarray(handle["var"]["gene_name"]).astype(str)
        axis = pd.Series(np.arange(len(all_genes)), index=all_genes)
        selected = axis.loc[gene_ids].to_numpy(int)
        sums = np.zeros((len(tasks), len(gene_ids)), dtype=np.float64)
        counts = np.zeros(len(tasks), dtype=np.int64)
        x = handle["X"]
        indptr = x["indptr"]
        for start in range(0, len(ids), chunk_rows):
            end = min(start + chunk_rows, len(ids))
            p0, p1 = int(indptr[start]), int(indptr[end])
            block = sparse.csr_matrix(
                (
                    np.asarray(x["data"][p0:p1]),
                    np.asarray(x["indices"][p0:p1]),
                    np.asarray(indptr[start : end + 1], dtype=np.int64) - p0,
                ),
                shape=(end - start, len(all_genes)),
            )[:, selected].toarray()
            accumulate(sums, counts, codes[start:end], block)
            if start % 100000 == 0:
                print(f"[McFalineConfirmationTruth] rows {start}:{end}/{len(ids)}", flush=True)
    if (counts == 0).any():
        raise RuntimeError(f"empty test truth aggregates: {int((counts == 0).sum())}")
    return sums / counts[:, None] - controls


def utility20(ids: np.ndarray, score: np.ndarray, truth: np.ndarray) -> float:
    if len(truth) < 20:
        return float("nan")
    k = int(math.ceil(0.2 * len(truth)))
    chosen = np.lexsort((ids.astype(str), -score))[:k]
    oracle = np.lexsort((ids.astype(str), -truth))[:k]
    denominator = float(truth[oracle].mean() - truth.mean())
    return float((truth[chosen].mean() - truth.mean()) / denominator) if denominator > 1e-12 else float("nan")


def metrics(ids: np.ndarray, score: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    order = np.argsort(score, kind="stable")
    result = {
        "utility20": utility20(ids, score, truth),
        "spearman": float(spearmanr(score, truth).statistic),
        "aurc": float(np.mean(np.cumsum(truth[order]) / np.arange(1, len(order) + 1))),
    }
    for coverage in (0.10, 0.20, 0.50):
        k = int(math.ceil(coverage * len(truth)))
        result[f"error_at_{int(coverage * 100)}"] = float(truth[order[:k]].mean())
    k = int(math.ceil(0.2 * len(truth)))
    high_truth = set(np.lexsort((ids.astype(str), -truth))[:k].tolist())
    high_score = set(np.lexsort((ids.astype(str), -score))[:k].tolist())
    result["high_risk_miss_rate"] = float(1.0 - len(high_truth & high_score) / k)
    return result


def macro_metrics(frame: pd.DataFrame, score_column: str) -> dict[str, float]:
    values = []
    for _, group in frame.groupby("stratum", sort=True):
        values.append(
            metrics(
                group.task_id.to_numpy(str),
                group[score_column].to_numpy(float),
                group.true_error_rmse.to_numpy(float),
            )
        )
    return {key: float(np.nanmean([row[key] for row in values])) for key in values[0]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    parser.add_argument("--risk-seal", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--chunk-rows", type=int, default=1000)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    tasks = pd.read_csv(args.aggregate_dir / "TEST_TASKS.csv").rename(
        columns={"condition": "perturbation", "cell_type": "context"}
    )
    tasks["task_id"] = (
        tasks.perturbation.astype(str)
        + "::"
        + tasks.context.astype(str)
        + "::"
        + tasks.treatment.astype(str)
    )
    tasks["stratum"] = tasks.context.astype(str) + "::" + tasks.treatment.astype(str)
    genes_payload = json.loads((args.aggregate_dir / "GENE_IDS.json").read_text())
    gene_ids = genes_payload["gene_ids"] if isinstance(genes_payload, dict) else genes_payload
    controls = np.load(args.aggregate_dir / "TEST_CONTROL_STATES.npy")
    predicted = np.load(args.aggregate_dir / "TEST_CALIBRATED_EFFECTS.npy")
    truth = aggregate_test_truth(
        args.h5ad, args.split, tasks, gene_ids, controls, args.chunk_rows
    )
    true_error = np.sqrt(np.mean(np.square(predicted - truth), axis=1))
    tasks["true_error_rmse"] = true_error
    np.save(args.output_dir / "TEST_TRUE_EFFECTS.npy", truth.astype(np.float32))
    tasks.to_csv(args.output_dir / "TEST_TASK_ERRORS.csv.gz", index=False)

    sealed = pd.read_csv(args.risk_seal / "SEALED_TEST_RISK_PREDICTIONS.csv.gz")
    wide = sealed.pivot(index="task_id", columns="method", values="risk")
    task = tasks.set_index("task_id").join(wide, how="left").reset_index()
    if task[sealed.method.unique()].isna().any().any():
        raise RuntimeError("sealed prediction/task alignment failed")

    strata_rows, macro_rows = [], []
    for method in sorted(sealed.method.unique()):
        for stratum, group in task.groupby("stratum", sort=True):
            row = metrics(
                group.task_id.to_numpy(str),
                group[method].to_numpy(float),
                group.true_error_rmse.to_numpy(float),
            )
            strata_rows.append(
                {"method": method, "stratum": stratum, "n_tasks": len(group), **row}
            )
        macro_rows.append({"method": method, **macro_metrics(task, method)})
    strata = pd.DataFrame(strata_rows)
    macro = pd.DataFrame(macro_rows)
    strata.to_csv(args.output_dir / "COLDSTART_STRATUM_RESULTS.csv", index=False)
    macro.to_csv(args.output_dir / "COLDSTART_MACRO_RESULTS.csv", index=False)

    baseline = "Magnitude"
    methods = [method for method in sorted(sealed.method.unique()) if method != baseline]
    perturbations = np.asarray(sorted(task.perturbation.unique()))
    cluster_rows = {
        value: np.flatnonzero(task.perturbation.to_numpy(str) == value)
        for value in perturbations
    }
    rng = np.random.default_rng(SEED)
    draws = {method: [] for method in methods}
    for _ in range(args.bootstrap):
        sampled = rng.integers(0, len(perturbations), len(perturbations))
        idx = np.concatenate([cluster_rows[perturbations[i]] for i in sampled])
        part = task.iloc[idx].copy()
        base_u20 = macro_metrics(part, baseline)["utility20"]
        for method in methods:
            draws[method].append(macro_metrics(part, method)["utility20"] - base_u20)
    bootstrap_rows = []
    for method, values in draws.items():
        values = np.asarray(values)
        point = (
            macro.loc[macro.method.eq(method), "utility20"].iloc[0]
            - macro.loc[macro.method.eq(baseline), "utility20"].iloc[0]
        )
        bootstrap_rows.append(
            {
                "method": method,
                "baseline": baseline,
                "delta_utility20": float(point),
                "bootstrap_mean_delta": float(np.nanmean(values)),
                "ci95_lower": float(np.nanquantile(values, 0.025)),
                "ci95_upper": float(np.nanquantile(values, 0.975)),
                "bootstrap_replicates": args.bootstrap,
            }
        )
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(args.output_dir / "COLDSTART_CLUSTER_BOOTSTRAP.csv", index=False)

    base_macro = macro.set_index("method").loc[baseline]
    gate_rows = []
    for method in ("ZeroLabelSharedHGB", "ValidationAdapted_LearnedPublic_HGB"):
        current = macro.set_index("method").loc[method]
        interval = bootstrap.set_index("method").loc[method]
        stratum_delta = (
            strata[strata.method.eq(method)].set_index("stratum").utility20
            - strata[strata.method.eq(baseline)].set_index("stratum").utility20
        )
        coverage_degradation = max(
            (current[f"error_at_{level}"] - base_macro[f"error_at_{level}"])
            / max(abs(base_macro[f"error_at_{level}"]), 1e-12)
            for level in (10, 20, 50)
        )
        aurc_degradation = (current.aurc - base_macro.aurc) / max(abs(base_macro.aurc), 1e-12)
        gate_rows.append(
            {
                "method": method,
                "delta_utility20": float(current.utility20 - base_macro.utility20),
                "ci95_lower": float(interval.ci95_lower),
                "nonnegative_strata_fraction": float((stratum_delta >= 0).mean()),
                "max_coverage_error_relative_degradation": float(coverage_degradation),
                "high_risk_miss_rate_degradation": float(
                    current.high_risk_miss_rate - base_macro.high_risk_miss_rate
                ),
                "aurc_relative_degradation": float(aurc_degradation),
                "passes_registered_gate": bool(
                    current.utility20 - base_macro.utility20 >= 0.005
                    and interval.ci95_lower >= -0.005
                    and (stratum_delta >= 0).mean() >= 0.60
                    and coverage_degradation <= 0.05
                    and current.high_risk_miss_rate - base_macro.high_risk_miss_rate <= 0.02
                    and aurc_degradation <= 0.05
                ),
            }
        )
    gates = pd.DataFrame(gate_rows)
    gates.to_csv(args.output_dir / "CONFIRMATION_GATE.csv", index=False)

    status = {
        "status": "COMPLETE",
        "test_truth_opened_once": True,
        "n_tasks": int(len(tasks)),
        "n_clusters": int(tasks.perturbation.nunique()),
        "n_strata": int(tasks.stratum.nunique()),
        "bootstrap_replicates": args.bootstrap,
        "primary_method": "ZeroLabelSharedHGB",
        "primary_gate_passed": bool(
            gates.set_index("method").loc["ZeroLabelSharedHGB", "passes_registered_gate"]
        ),
        "secondary_method": "ValidationAdapted_LearnedPublic_HGB",
        "secondary_gate_passed": bool(
            gates.set_index("method").loc[
                "ValidationAdapted_LearnedPublic_HGB", "passes_registered_gate"
            ]
        ),
    }
    (args.output_dir / "RUN_STATUS.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    print(macro.to_string(index=False), flush=True)
    print(bootstrap.to_string(index=False), flush=True)
    print(gates.to_string(index=False), flush=True)
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
