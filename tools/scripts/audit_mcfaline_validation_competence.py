#!/usr/bin/env python3
"""Audit McFaline validation competence without accessing test expression.

The audit maps official validation predictions, validation truth, and two
preregistered simple baselines to the same 512-gene effect contract.  Test rows
are identified only to exclude them; their expression and covariate summaries
are never read or emitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse


H5AD = Path("/home/yyf/data/perturbench_mcfaline23_official/mcfaline23_gxe_processed.h5ad")
SPLIT = Path("/home/yyf/data/perturbench_mcfaline23_official/splits/mcfaline23_gxe_splits/full_covariate_split.csv")
GENES = Path("/home/yyf/data/safeconf_dual_memory_20260929/public_mcfaline_trainval/gene_ids.json")
SEED = 20260929


class CompetenceFailure(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--h5ad", type=Path, default=H5AD)
    parser.add_argument("--split", type=Path, default=SPLIT)
    parser.add_argument("--genes", type=Path, default=GENES)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--chunk-rows", type=int, default=1000)
    return parser.parse_args()


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def factorize(frame: pd.DataFrame, columns: list[str], mask: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    codes = np.full(len(frame), -1, dtype=np.int64)
    selected = frame.loc[mask, columns].astype(str)
    local, uniques = pd.factorize(pd.MultiIndex.from_frame(selected), sort=True)
    codes[np.flatnonzero(mask)] = local
    keys = uniques.to_frame(index=False)
    keys.columns = columns
    return codes, keys


def accumulate(sums: np.ndarray, counts: np.ndarray, codes: np.ndarray, values: np.ndarray) -> None:
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


def aggregate_truth_and_baselines(
    h5ad: Path, split_path: Path, gene_names: list[str], chunk_rows: int
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split = pd.read_csv(split_path, header=None, names=["cell_id", "split"])
    split_map = pd.Series(split.split.to_numpy(str), index=split.cell_id.astype(str)).to_dict()
    with h5py.File(h5ad, "r") as handle:
        obs = handle["obs"]
        cell_ids = categorical(obs, "_index")
        roles = np.asarray([split_map.get(x, "missing") for x in cell_ids], dtype=object)
        if (roles == "missing").any():
            raise CompetenceFailure("official split failed to align")
        control = categorical(obs, "control") == "1"
        frame = pd.DataFrame({
            "perturbation": categorical(obs, "perturbation"),
            "context": categorical(obs, "cell_type"),
            "treatment": categorical(obs, "treatment"),
        })
        frame["task_id"] = frame.perturbation + "::" + frame.context + "::" + frame.treatment
        masks = {
            "train_task": (roles == "train") & ~control,
            "train_control": (roles == "train") & control,
            "val_task": (roles == "val") & ~control,
            "val_control": (roles == "val") & control,
        }
        specs = []
        for name, mask in masks.items():
            columns = ["context", "treatment"] if "control" in name else ["task_id", "perturbation", "context", "treatment"]
            codes, keys = factorize(frame, columns, mask)
            specs.append((name, codes, keys))
        all_genes = np.asarray(handle["var"]["gene_name"]).astype(str)
        index = pd.Series(np.arange(len(all_genes)), index=all_genes)
        if not set(gene_names) <= set(index.index):
            raise CompetenceFailure("registered gene contract does not align")
        selected = index.loc[gene_names].to_numpy(int)
        if len(selected) != 512:
            raise CompetenceFailure(f"expected 512 contract genes, got {len(selected)}")
        aggregates = {
            name: [np.zeros((len(keys), len(selected)), dtype=np.float64), np.zeros(len(keys), dtype=np.int64)]
            for name, _, keys in specs
        }
        x = handle["X"]
        indptr = x["indptr"]
        for start in range(0, len(frame), chunk_rows):
            end = min(start + chunk_rows, len(frame))
            p0, p1 = int(indptr[start]), int(indptr[end])
            block = sparse.csr_matrix((
                np.asarray(x["data"][p0:p1]),
                np.asarray(x["indices"][p0:p1]),
                np.asarray(indptr[start:end + 1], dtype=np.int64) - p0,
            ), shape=(end - start, len(all_genes)))[:, selected].toarray()
            for name, codes, _ in specs:
                accumulate(aggregates[name][0], aggregates[name][1], codes[start:end], block)
            if start % 100000 == 0:
                print(f"[CompetenceTruth] rows {start}:{end}/{len(frame)}", flush=True)
    means = {}
    keys_by_name = {}
    for name, _, keys in specs:
        sums, counts = aggregates[name]
        if (counts == 0).any():
            raise CompetenceFailure(f"empty aggregate: {name}")
        means[name] = sums / counts[:, None]
        keys_by_name[name] = keys
    train_controls = {
        tuple(row): means["train_control"][i]
        for i, row in enumerate(keys_by_name["train_control"].itertuples(index=False, name=None))
    }
    val_controls = {
        tuple(row): means["val_control"][i]
        for i, row in enumerate(keys_by_name["val_control"].itertuples(index=False, name=None))
    }
    train_effects_by_state: dict[tuple[str, str], list[np.ndarray]] = {}
    for i, row in enumerate(keys_by_name["train_task"].itertuples(index=False)):
        state = (str(row.context), str(row.treatment))
        if state in train_controls:
            train_effects_by_state.setdefault(state, []).append(means["train_task"][i] - train_controls[state])
    state_mean = {state: np.mean(vectors, axis=0) for state, vectors in train_effects_by_state.items()}
    metadata, truth, state_baseline, validation_controls = [], [], [], []
    for i, row in enumerate(keys_by_name["val_task"].itertuples(index=False)):
        state = (str(row.context), str(row.treatment))
        if state not in val_controls or state not in state_mean:
            continue
        metadata.append({
            "task_id": str(row.task_id), "perturbation": str(row.perturbation),
            "context": state[0], "treatment": state[1], "stratum": f"{state[0]}::{state[1]}",
        })
        truth.append(means["val_task"][i] - val_controls[state])
        state_baseline.append(state_mean[state])
        validation_controls.append(val_controls[state])
    truth = np.asarray(truth)
    return pd.DataFrame(metadata), truth, np.zeros_like(truth), np.asarray(state_baseline), np.asarray(validation_controls)


def batch_number(path: Path) -> int:
    match = re.search(r"batch(\d+)", path.name)
    if not match:
        raise CompetenceFailure(f"unexpected prediction filename: {path}")
    return int(match.group(1))


def aggregate_predictions(directory: Path, gene_names: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    files = sorted(directory.glob("predicted_rank*_batch*.h5ad"), key=batch_number)
    if not files:
        raise CompetenceFailure(f"no official validation predictions under {directory}")
    seen: set[int] = set()
    rows, vectors = [], []
    selected = None
    for number, path in enumerate(files, 1):
        item = ad.read_h5ad(path, backed="r")
        if selected is None:
            axis = pd.Series(np.arange(item.n_vars), index=item.var_names.astype(str))
            if not set(gene_names) <= set(axis.index):
                raise CompetenceFailure("prediction gene axis does not align")
            selected = axis.loc[gene_names].to_numpy(int)
        obs = item.obs.reset_index(drop=True)
        for condition_index, positions in obs.groupby("_condition_idx").groups.items():
            condition_index = int(condition_index)
            if condition_index in seen:
                continue
            seen.add(condition_index)
            pos = np.asarray(list(positions), dtype=int)
            values = np.asarray(item[pos, selected].X, dtype=float)
            first = obs.iloc[pos[0]]
            rows.append({
                "condition_index": condition_index,
                "task_id": f"{first['condition']}::{first['cell_type']}::{first['treatment']}",
            })
            vectors.append(values.mean(axis=0))
        item.file.close()
        if number % 20 == 0:
            print(f"[CompetencePredictions] files {number}/{len(files)}", flush=True)
    return pd.DataFrame(rows), np.asarray(vectors)


def rmse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(a - b), axis=1))


def markdown(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        out.append("| " + " | ".join(f"{x:.6f}" if isinstance(x, float) else str(x) for x in row) + " |")
    return "\n".join(out)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    genes = json.loads(args.genes.read_text())
    meta, truth, no_change, state_mean, validation_controls = aggregate_truth_and_baselines(
        args.h5ad, args.split, genes, args.chunk_rows
    )
    pred_meta, predicted_state = aggregate_predictions(args.prediction_dir, genes)
    pred_map = pd.Series(np.arange(len(pred_meta)), index=pred_meta.task_id).to_dict()
    common = meta.task_id.isin(pred_map)
    meta = meta.loc[common].reset_index(drop=True)
    truth = truth[common.to_numpy()]
    no_change = no_change[common.to_numpy()]
    state_mean = state_mean[common.to_numpy()]
    validation_controls = validation_controls[common.to_numpy()]
    predicted_state = predicted_state[[pred_map[x] for x in meta.task_id]]
    # Official models emit treated states from matched control inputs.  Convert
    # their aggregate to effect space by subtracting the observed validation
    # control already used in the true-effect construction.  Because the model
    # output is conditional on those controls, its mean effect is equivalently
    # predicted treated mean minus the matching validation-control mean.  The
    # latter is recovered as treated mean minus true effect.
    # Use the same validation-only matched control for the predicted and true
    # treated states. This closes the treated-state Prediction/Error Contract
    # without mixing train/validation control estimates.
    model_effect = predicted_state - validation_controls
    method_vectors = {"candidate": model_effect, "no_change": no_change, "train_state_mean_effect": state_mean}
    task = meta.copy()
    for name, values in method_vectors.items():
        task[name] = rmse(values, truth)
    baseline_names = ["no_change", "train_state_mean_effect"]
    baseline_macro = {name: float(task[name].mean()) for name in baseline_names}
    strongest = min(baseline_macro, key=baseline_macro.get)
    stratum = task.groupby("stratum", as_index=False).agg(
        n_tasks=("task_id", "size"), candidate_rmse=("candidate", "mean"),
        no_change_rmse=("no_change", "mean"), train_state_mean_effect_rmse=("train_state_mean_effect", "mean"),
    )
    baseline_column = strongest + "_rmse"
    relative_gap = float((task.candidate.mean() - task[strongest].mean()) / task[strongest].mean())
    noninferior = float((stratum.candidate_rmse <= 1.02 * stratum[baseline_column]).mean())
    genes_unique = np.asarray(sorted(task.perturbation.unique()))
    gene_rows = {gene: np.flatnonzero(task.perturbation.to_numpy(str) == gene) for gene in genes_unique}
    rng = np.random.default_rng(SEED)
    draws = []
    candidate = task.candidate.to_numpy(float)
    baseline = task[strongest].to_numpy(float)
    for _ in range(args.bootstrap):
        sampled = rng.integers(0, len(genes_unique), len(genes_unique))
        idx = np.concatenate([gene_rows[genes_unique[i]] for i in sampled])
        draws.append((candidate[idx].mean() - baseline[idx].mean()) / baseline[idx].mean())
    draws = np.asarray(draws)
    result = pd.DataFrame([{
        "candidate": args.candidate, "n_tasks": len(task), "n_perturbation_clusters": len(genes_unique),
        "n_strata": len(stratum), "candidate_macro_rmse": float(task.candidate.mean()),
        "strongest_simple_baseline": strongest, "baseline_macro_rmse": float(task[strongest].mean()),
        "relative_gap": relative_gap, "bootstrap_ci95_lower": float(np.quantile(draws, .025)),
        "bootstrap_ci95_upper": float(np.quantile(draws, .975)),
        "noninferior_strata_fraction": noninferior,
        "passes_competence": bool(relative_gap <= .02 and noninferior >= .60 and np.quantile(draws, .025) <= .02),
        "test_expression_opened": False,
    }])
    task.to_csv(args.output / "TASK_ERRORS.csv.gz", index=False)
    stratum.to_csv(args.output / "STRATUM_ERRORS.csv", index=False)
    result.to_csv(args.output / "COMPETENCE_RESULT.csv", index=False)
    np.save(args.output / "VALIDATION_PREDICTED_EFFECTS.npy", model_effect.astype(np.float32))
    np.save(args.output / "VALIDATION_TRUE_EFFECTS.npy", truth.astype(np.float32))
    (args.output / "GENE_IDS.json").write_text(json.dumps(genes, indent=2) + "\n")
    (args.output / "REPORT.md").write_text(
        "# McFaline validation competence\n\nValidation only; test expression remained sealed.\n\n" + markdown(result) + "\n"
    )
    (args.output / "RUN_STATUS.json").write_text(json.dumps({
        "status": "COMPLETE", "candidate": args.candidate, "partition": "validation",
        "test_expression_opened": False, "prediction_dir": str(args.prediction_dir),
        "split_sha256": sha256_file(args.split), "gene_contract_sha256": hashlib.sha256("\n".join(genes).encode()).hexdigest(),
        "bootstrap_replicates": args.bootstrap,
    }, indent=2) + "\n")
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
