#!/usr/bin/env python3
"""Transfer a frozen risk learner to an upstream with zero target error labels.

For each direction, the learner, imputation, scaling, and error CDF are fitted
only on one source upstream.  The target upstream contributes no error label to
training.  Source rows from the target biological fold are also excluded so a
shared experimental truth cannot cross the fold through another architecture.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.scripts.run_dual_memory_txpert_public_biology import (
    P,
    PRIOR_FEATURES,
    SEED,
    TARGETS,
    aurc,
    fit_predict_regressor,
    rank_training_labels,
    rho,
    utility20,
)


DEFAULT_INPUT = ROOT / "docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/txpert_public_biology_repaired/SHARED_RISK_FEATURES.csv.gz"
DEFAULT_OUTPUT = ROOT / "docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/cross_predictor_zero_label_transfer"
METHODS = {
    "Ridge_P": (P, "ridge"),
    "Ridge_ManualPublic": (P + [f"CellWeighted_{x}" for x in PRIOR_FEATURES], "ridge"),
    "Ridge_RepairedPublic": (P + [f"LearnedHGBRegularized_{x}" for x in PRIOR_FEATURES], "ridge"),
    "HGB_P": (P, "hgb"),
    "HGB_ManualPublic": (P + [f"CellWeighted_{x}" for x in PRIOR_FEATURES], "hgb"),
    "HGB_RepairedPublic": (P + [f"LearnedHGBRegularized_{x}" for x in PRIOR_FEATURES], "hgb"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=5000)
    return parser.parse_args()


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    compression = {"method": "gzip", "mtime": 0} if path.name.endswith(".gz") else None
    frame.to_csv(temporary, index=False, compression=compression)
    os.replace(temporary, path)


def predict(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    upstreams = sorted(frame.upstream.unique())
    if len(upstreams) != 2:
        raise ValueError(f"expected two upstreams, got {upstreams}")
    for source, target in ((upstreams[0], upstreams[1]), (upstreams[1], upstreams[0])):
        for fold in sorted(frame.fold.unique()):
            fit = frame[frame.upstream.eq(source) & frame.fold.ne(fold)].copy().reset_index(drop=True)
            query = frame[frame.upstream.eq(target) & frame.fold.eq(fold)].copy().reset_index(drop=True)
            if fit.empty or query.empty or fit.upstream.nunique() != 1 or query.upstream.nunique() != 1:
                raise RuntimeError("source/target isolation failed")
            if set(fit.task_id).intersection(query.task_id):
                raise RuntimeError("same biological task crossed transfer train/test")
            labels = rank_training_labels(fit)
            for method, (columns, learner) in METHODS.items():
                score = np.clip(fit_predict_regressor(fit, query, columns, labels, learner), 0, 1)
                for row, value in zip(query.itertuples(), score):
                    rows.append({
                        "source_upstream": source,
                        "target_upstream": target,
                        "task_id": row.task_id,
                        "target": row.target,
                        "gene": row.gene,
                        "fold": int(fold),
                        "method": method,
                        "predicted_risk": float(value),
                        "true_error_rmse": float(row.true_error_rmse),
                        "target_error_labels_used_for_fit": False,
                    })
            for row in query.itertuples():
                rows.append({
                    "source_upstream": source,
                    "target_upstream": target,
                    "task_id": row.task_id,
                    "target": row.target,
                    "gene": row.gene,
                    "fold": int(fold),
                    "method": "Magnitude",
                    "predicted_risk": float(row.predicted_magnitude),
                    "true_error_rmse": float(row.true_error_rmse),
                    "target_error_labels_used_for_fit": False,
                })
    return pd.DataFrame(rows)


def summaries(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    strata = []
    for keys, group in predictions.groupby(
        ["source_upstream", "target_upstream", "target", "method"], sort=True
    ):
        source, target_upstream, context, method = keys
        strata.append({
            "source_upstream": source,
            "target_upstream": target_upstream,
            "target": context,
            "method": method,
            "n_tasks": len(group),
            "utility20": utility20(group.task_id.to_numpy(str), group.predicted_risk.to_numpy(float), group.true_error_rmse.to_numpy(float)),
            "spearman": rho(group.predicted_risk.to_numpy(float), group.true_error_rmse.to_numpy(float)),
            "aurc": aurc(group.predicted_risk.to_numpy(float), group.true_error_rmse.to_numpy(float)),
        })
    strata_frame = pd.DataFrame(strata)
    summary = strata_frame.groupby(
        ["source_upstream", "target_upstream", "method"], as_index=False
    ).agg(
        n_strata=("target", "nunique"),
        utility20=("utility20", "mean"),
        spearman=("spearman", "mean"),
        aurc=("aurc", "mean"),
    )
    return strata_frame, summary


def bootstrap(predictions: pd.DataFrame, replicates: int) -> pd.DataFrame:
    comparisons = [
        ("HGBRepaired_vs_Magnitude", "HGB_RepairedPublic", "Magnitude"),
        ("HGBRepaired_vs_HGBP", "HGB_RepairedPublic", "HGB_P"),
        ("HGBRepaired_vs_HGBManual", "HGB_RepairedPublic", "HGB_ManualPublic"),
        ("RidgeRepaired_vs_Magnitude", "Ridge_RepairedPublic", "Magnitude"),
        ("RidgeRepaired_vs_RidgeP", "Ridge_RepairedPublic", "Ridge_P"),
        ("RidgeRepaired_vs_RidgeManual", "Ridge_RepairedPublic", "Ridge_ManualPublic"),
    ]
    wide = predictions.pivot(
        index=["source_upstream", "target_upstream", "task_id", "target", "gene", "true_error_rmse"],
        columns="method", values="predicted_risk",
    ).reset_index()
    rng = np.random.default_rng(SEED + 17)
    output = []
    for (source, target_upstream), part in wide.groupby(["source_upstream", "target_upstream"], sort=True):
        genes = np.asarray(sorted(part.gene.astype(str).unique()))
        gene_rows = {g: np.flatnonzero(part.gene.astype(str).to_numpy() == g) for g in genes}
        ids = part.task_id.to_numpy(str)
        contexts = part.target.to_numpy(str)
        truth = part.true_error_rmse.to_numpy(float)
        arrays = {m: part[m].to_numpy(float) for m in predictions.method.unique()}
        draws = {name: [] for name, _, _ in comparisons}
        for _ in range(replicates):
            sampled = rng.integers(0, len(genes), len(genes))
            idx = np.concatenate([gene_rows[genes[i]] for i in sampled])
            metrics = {}
            for method, values in arrays.items():
                per_context = []
                for context in TARGETS:
                    use = idx[contexts[idx] == context]
                    per_context.append(utility20(ids[use], values[use], truth[use]))
                metrics[method] = float(np.nanmean(per_context))
            for name, first, second in comparisons:
                draws[name].append(metrics[first] - metrics[second])
        for name, first, second in comparisons:
            values = np.asarray(draws[name])
            output.append({
                "source_upstream": source,
                "target_upstream": target_upstream,
                "comparison": name,
                "method_a": first,
                "method_b": second,
                "delta_utility20": float(values.mean()),
                "ci95_lower": float(np.quantile(values, 0.025)),
                "ci95_upper": float(np.quantile(values, 0.975)),
                "bootstrap_replicates": replicates,
            })
    return pd.DataFrame(output)


def markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    frame = pd.read_csv(args.input)
    if frame.groupby("task_id").fold.nunique().max() != 1:
        raise RuntimeError("same biological task does not share one fold")
    args.output.mkdir(parents=True)
    predictions = predict(frame)
    strata, summary = summaries(predictions)
    intervals = bootstrap(predictions, args.bootstrap)
    atomic_csv(args.output / "ZERO_LABEL_PREDICTIONS.csv.gz", predictions)
    atomic_csv(args.output / "ZERO_LABEL_STRATA.csv", strata)
    atomic_csv(args.output / "ZERO_LABEL_SUMMARY.csv", summary)
    atomic_csv(args.output / "ZERO_LABEL_BOOTSTRAP.csv", intervals)
    status = {
        "status": "COMPLETE",
        "evidence_role": "DEV_SEEN",
        "n_tasks_per_target_upstream": int(predictions.task_id.nunique()),
        "n_directions": 2,
        "outer_folds": int(predictions.fold.nunique()),
        "bootstrap_replicates": args.bootstrap,
        "target_upstream_error_labels_used_for_fit": False,
        "source_same_biological_fold_excluded": True,
        "mcfaline_test_opened": False,
    }
    (args.output / "RUN_STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
    report = [
        "# Zero-target-error cross-predictor transfer", "",
        "The source upstream supplies all risk labels. The target upstream supplies zero error labels to fitting, preprocessing, scaling or error-CDF construction. Source rows from the queried biological fold are excluded.", "",
        "## Summary", "", markdown(summary), "",
        "## Paired perturbation-cluster bootstrap", "", markdown(intervals), "",
    ]
    (args.output / "REPORT.md").write_text("\n".join(report))
    print(summary.to_string(index=False), flush=True)
    print(intervals.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
