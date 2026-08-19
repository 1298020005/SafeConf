#!/usr/bin/env python3
"""Freeze E201 tasks and source-only evidence before target predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime
from pathlib import Path

import anndata as ad
import joblib
import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
OUT = ROOT / "docs/实验结果/E201_txpert_multitarget_retraining_20260802"
FREEZE = OUT / "PRETRUTH_TASK_BASE_FREEZE.md"
TABLES = OUT / "tables"
STATUS = OUT / "E201_PRETRUTH_TASK_BASE_STATUS.json"
TARGETS = ("K562", "RPE1", "hepg2", "jurkat")
EXPECTED_TASKS = {
    "K562": (580, 80_153, 566, 14),
    "RPE1": (467, 38_543, 416, 51),
    "hepg2": (480, 30_139, 405, 75),
    "jurkat": (481, 43_604, 421, 60),
}
PREDICTION_VIEW_SHA256 = (
    "85f93d1b29ded34d9dcece9ecdba1ef722a3f14aeedbfbe740eed9f045fbe486"
)
TRAINING_VIEW_SHA256 = {
    "K562": "8a19d3d4048800c06827e2f28e983bfa6f67b1945d2081692ce3d69a45d471db",
    "RPE1": "9aeec2bdc56461713e9039348428d63aa1b98f6000835ed3253c35d6d85a387d",
    "hepg2": "1f0dc20806bd40cd151ebfebb59a9fdac5ad14c4e223a655ae0ed6de890ed891",
    "jurkat": "5a944ec0f114e2398f2058072121d130deee5af1f97def926619f5ee30c231fb",
}
TRAINING_MANIFEST_SHA256 = {
    "K562": "8114be7febf1f166da729cf23cb9dc2356d7e54b6f255581b665997d7651265d",
    "RPE1": "e2f1e6aaa91a5e4143bbacf578453eaeaa135dc6e0340921d048de5a20ce9202",
    "hepg2": "cf3b0ce0a93d669c7963b2198ce27dcf053249f1af0e0928777a4de4d84e2873",
    "jurkat": "fa9bc3cf9d22f8df2635d3932402db9cc455826f5226aceb366b64a19a4ddb21",
}
GEARS_GENE_SET_SHA256 = (
    "7e2be69a204b72349b793cc6723a5f88419f1ca6472ea5e28c5f7d623ee8e23d"
)


class TaskBaseFailure(RuntimeError):
    pass


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--txpert-repo", type=Path, required=True)
    parser.add_argument("--vector-output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_text(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def tracked_clean(path: Path) -> bool:
    relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
    commands = (
        ["git", "-C", str(ROOT), "cat-file", "-e", f"HEAD:{relative}"],
        ["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", relative],
        [
            "git",
            "-C",
            str(ROOT),
            "diff",
            "--cached",
            "--quiet",
            "HEAD",
            "--",
            relative,
        ],
    )
    return all(
        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
        for command in commands
    )


def verify_git_release() -> str:
    if not tracked_clean(SCRIPT) or not tracked_clean(FREEZE):
        raise TaskBaseFailure("task-base runner/freeze is not tracked and clean")
    branch = git_text("branch", "--show-current")
    head = git_text("rev-parse", "HEAD")
    if not branch:
        raise TaskBaseFailure("detached HEAD is not allowed")
    for remote in ("origin", "github"):
        tracking = git_text("rev-parse", f"{remote}/{branch}")
        if tracking != head:
            raise TaskBaseFailure(f"{remote}/{branch} differs from local HEAD")
    return head


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(temporary, path)


def row_mean(matrix, indices: np.ndarray) -> np.ndarray:
    if len(indices) == 0:
        raise TaskBaseFailure("cannot average an empty matrix block")
    return np.asarray(matrix[indices].mean(axis=0), dtype=np.float64).ravel()


def source_evidence(
    target: str,
    cache: Path,
    conditions: set[str],
) -> tuple[dict[str, dict[str, np.ndarray]], pd.DataFrame, pd.DataFrame]:
    h5ad_path = cache / "de_adata_test.h5ad"
    manifest_path = cache / "E201_BLIND_VIEW_MANIFEST.json"
    if (
        not h5ad_path.is_file()
        or sha256_file(h5ad_path) != TRAINING_VIEW_SHA256[target]
        or not manifest_path.is_file()
        or sha256_file(manifest_path) != TRAINING_MANIFEST_SHA256[target]
    ):
        raise TaskBaseFailure(f"blind training cache changed: {target}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("target") != target
        or int(manifest.get("n_target_treatments", -1)) != 0
        or manifest.get("uns_keys") != []
    ):
        raise TaskBaseFailure(f"blind training manifest failed: {target}")

    dataset = ad.read_h5ad(h5ad_path, backed="r")
    obs = dataset.obs.copy()
    is_control = obs.control.astype(bool)
    if int((~is_control & obs.cell_line.astype(str).eq(target)).sum()) != 0:
        dataset.file.close()
        raise TaskBaseFailure(f"target perturbation survived physical view: {target}")
    deltas: dict[str, dict[str, np.ndarray]] = {
        condition: {} for condition in conditions
    }
    support_rows = []
    access_rows = []
    try:
        for context in sorted(set(TARGETS) - {target}):
            context_mask = obs.cell_line.astype(str).eq(context)
            control_global = np.flatnonzero((context_mask & is_control).to_numpy())
            perturb_global = np.flatnonzero(
                (
                    context_mask
                    & ~is_control
                    & obs.condition.astype(str).isin(conditions)
                ).to_numpy()
            )
            if not len(control_global) or not len(perturb_global):
                raise TaskBaseFailure(f"empty source block: {target}/{context}")
            control_x = sparse.csr_matrix(dataset.X[control_global])
            perturb_x = sparse.csr_matrix(dataset.X[perturb_global])
            control_obs = obs.iloc[control_global].reset_index(drop=True)
            perturb_obs = obs.iloc[perturb_global].reset_index(drop=True)
            control_batches = control_obs.batch.astype(str).to_numpy()
            control_means = {
                batch: row_mean(control_x, np.flatnonzero(control_batches == batch))
                for batch in sorted(pd.unique(control_batches))
            }
            condition_values = perturb_obs.condition.astype(str).to_numpy()
            perturb_batches = perturb_obs.batch.astype(str).to_numpy()
            for condition in sorted(pd.unique(condition_values)):
                local = np.flatnonzero(condition_values == condition)
                used_batches, batch_counts = np.unique(
                    perturb_batches[local], return_counts=True
                )
                missing = [
                    batch for batch in used_batches if batch not in control_means
                ]
                if missing:
                    raise TaskBaseFailure(
                        f"missing matched control: {target}/{context}/{condition}/{missing}"
                    )
                matched_control = sum(
                    control_means[batch] * int(count)
                    for batch, count in zip(used_batches, batch_counts)
                ) / len(local)
                delta = row_mean(perturb_x, local) - matched_control
                if not np.isfinite(delta).all():
                    raise TaskBaseFailure(
                        f"non-finite source delta: {target}/{context}/{condition}"
                    )
                deltas[condition][context] = delta
                support_rows.append(
                    {
                        "target": target,
                        "condition": condition,
                        "source_context": context,
                        "n_source_perturbed_cells": len(local),
                        "n_source_batches": len(used_batches),
                        "n_source_control_cells": len(control_obs),
                    }
                )
            access_rows.extend(
                [
                    {
                        "target": target,
                        "source_context": context,
                        "row_kind": "source_control",
                        "n_rows": len(control_global),
                        "target_perturbed_expression_rows": 0,
                    },
                    {
                        "target": target,
                        "source_context": context,
                        "row_kind": "source_train_perturbation",
                        "n_rows": len(perturb_global),
                        "target_perturbed_expression_rows": 0,
                    },
                ]
            )
    finally:
        dataset.file.close()
    missing_conditions = sorted(
        condition for condition, context_map in deltas.items() if not context_map
    )
    if missing_conditions:
        raise TaskBaseFailure(
            f"tasks without source evidence for {target}: {missing_conditions[:10]}"
        )
    return deltas, pd.DataFrame(support_rows), pd.DataFrame(access_rows)


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    txpert_repo = args.txpert_repo.resolve()
    vector_output = args.vector_output.resolve()
    repo_outputs = (
        STATUS,
        TABLES / "E201_PRETRUTH_TASK_BASE.csv",
        TABLES / "E201_SOURCE_CONTEXT_SUPPORT.csv",
        TABLES / "E201_SOURCE_EXPRESSION_ACCESS_AUDIT.csv",
    )
    if vector_output.exists() or any(path.exists() for path in repo_outputs):
        raise TaskBaseFailure("task-base output already exists")
    safeconf_commit = verify_git_release()

    cache_root = data_root / "txpert_official_20260802/cache"
    prediction_cache = cache_root / "E201_prediction_blind"
    prediction_h5ad = prediction_cache / "de_adata_test.h5ad"
    split_path = prediction_cache / "splits/train_test_split.pkl"
    subgroup_path = prediction_cache / "splits/subgroup.pkl"
    gene_set_path = txpert_repo / "data/gears_gene_set.csv"
    if (
        not prediction_h5ad.is_file()
        or sha256_file(prediction_h5ad) != PREDICTION_VIEW_SHA256
        or not gene_set_path.is_file()
        or sha256_file(gene_set_path) != GEARS_GENE_SET_SHA256
    ):
        raise TaskBaseFailure("prediction view or perturbation vocabulary changed")
    split = joblib.load(split_path)
    subgroup = joblib.load(subgroup_path)["test_subgroup"]
    train_conditions = set(map(str, split["train"]))
    test_conditions = set(map(str, split["test"]))
    context_only = set(map(str, subgroup["unseen_cell"])) & train_conditions
    perturbation_vocabulary = set(
        pd.read_csv(gene_set_path, index_col=0)["0"].astype(str)
    ) | {"ctrl"}

    prediction_data = ad.read_h5ad(prediction_h5ad, backed="r")
    prediction_obs = prediction_data.obs.copy()
    prediction_data.file.close()
    task_blocks = []
    support_blocks = []
    access_blocks = []
    mean_delta_blocks = []
    vector_row = 0
    for target in TARGETS:
        target_rows = prediction_obs.loc[
            prediction_obs.cell_line.astype(str).eq(target)
            & ~prediction_obs.control.astype(bool)
            & prediction_obs.condition.astype(str).isin(test_conditions)
        ].copy()
        token_valid = target_rows.condition.astype(str).map(
            lambda value: all(
                token in perturbation_vocabulary for token in value.split("+")
            )
        )
        target_rows = target_rows.loc[token_valid]
        available = set(target_rows.condition.astype(str))
        strict = available & context_only
        counts = (
            target_rows.loc[target_rows.condition.astype(str).isin(strict)]
            .groupby("condition", observed=True)
            .agg(
                n_target_cells=("condition", "size"),
                n_target_batches=("batch", "nunique"),
            )
            .sort_index()
        )
        expected = EXPECTED_TASKS[target]
        observed = (
            len(counts),
            int(counts.n_target_cells.sum()),
            int(counts.n_target_cells.ge(30).sum()),
            int(counts.n_target_cells.between(10, 29).sum()),
        )
        if observed != expected or int(counts.n_target_cells.lt(10).sum()) != 0:
            raise TaskBaseFailure(
                f"blind task inventory changed for {target}: {observed}"
            )

        deltas, support, access = source_evidence(
            target,
            cache_root / f"E201_blind_{target}",
            set(counts.index.astype(str)),
        )
        support_blocks.append(support)
        access_blocks.append(access)
        target_task_rows = []
        target_vectors = []
        observed_dispersions = []
        for condition, count_row in counts.iterrows():
            context_map = deltas[str(condition)]
            members = np.stack(list(context_map.values()), axis=0)
            center = members.mean(axis=0)
            dispersion = (
                float(np.sqrt(np.mean(np.square(members - center[None, :]))))
                if len(members) >= 2
                else math.nan
            )
            if math.isfinite(dispersion):
                observed_dispersions.append(dispersion)
            condition_support = support.loc[support.condition.eq(condition)]
            target_task_rows.append(
                {
                    "task_id": f"{target}::{condition}",
                    "target": target,
                    "condition": condition,
                    "gene": str(condition).split("+")[0],
                    "n_target_cells": int(count_row.n_target_cells),
                    "n_target_batches": int(count_row.n_target_batches),
                    "n_source_cells": int(
                        condition_support.n_source_perturbed_cells.sum()
                    ),
                    "n_source_contexts": len(context_map),
                    "source_delta_dispersion_observed": dispersion,
                    "dispersion_imputed": len(context_map) == 1,
                    "negative_log_source_cells": -math.log1p(
                        int(condition_support.n_source_perturbed_cells.sum())
                    ),
                    "support_context_deficit": 3 - len(context_map),
                    "analysis_stratum": (
                        "primary_ge30"
                        if int(count_row.n_target_cells) >= 30
                        else "sensitivity_10_29"
                    ),
                    "source_mean_delta_row": vector_row,
                }
            )
            target_vectors.append(center.astype(np.float32))
            vector_row += 1
        if not observed_dispersions:
            raise TaskBaseFailure(f"no observed dispersion values for {target}")
        imputation = float(np.median(observed_dispersions))
        target_frame = pd.DataFrame(target_task_rows)
        target_frame["source_delta_dispersion"] = target_frame[
            "source_delta_dispersion_observed"
        ].fillna(imputation)
        target_frame["dispersion_imputation_value"] = imputation
        task_blocks.append(target_frame)
        mean_delta_blocks.append(np.stack(target_vectors))

    tasks = pd.concat(task_blocks, ignore_index=True)
    support = pd.concat(support_blocks, ignore_index=True)
    access = pd.concat(access_blocks, ignore_index=True)
    mean_deltas = np.concatenate(mean_delta_blocks, axis=0).astype(np.float32)
    if (
        len(tasks) != 2_008
        or tasks.task_id.nunique() != 2_008
        or int(tasks.analysis_stratum.eq("primary_ge30").sum()) != 1_808
        or mean_deltas.shape != (2_008, 3_352)
        or int(access.target_perturbed_expression_rows.sum()) != 0
        or not np.isfinite(mean_deltas).all()
    ):
        raise TaskBaseFailure("combined task-base contract failed")
    if not np.array_equal(
        tasks.source_mean_delta_row.to_numpy(), np.arange(len(tasks))
    ):
        raise TaskBaseFailure("source vector row order changed")

    atomic_npy(vector_output, mean_deltas)
    atomic_csv(TABLES / "E201_PRETRUTH_TASK_BASE.csv", tasks)
    atomic_csv(TABLES / "E201_SOURCE_CONTEXT_SUPPORT.csv", support)
    atomic_csv(TABLES / "E201_SOURCE_EXPRESSION_ACCESS_AUDIT.csv", access)
    status = {
        "experiment": "E201_txpert_multitarget_retraining",
        "stage": "PRETRUTH_TASK_BASE",
        "status": "PASS",
        "generated_at": now(),
        "safeconf_commit": safeconf_commit,
        "n_tasks": len(tasks),
        "n_primary_tasks": int(tasks.analysis_stratum.eq("primary_ge30").sum()),
        "n_sensitivity_tasks": int(
            tasks.analysis_stratum.eq("sensitivity_10_29").sum()
        ),
        "n_source_support_rows": len(support),
        "n_source_access_records": len(access),
        "target_perturbed_expression_rows_opened": 0,
        "target_predictions_opened": False,
        "target_outcomes_evaluated": False,
        "source_mean_delta_file": {
            "path": "DATA/" + vector_output.relative_to(data_root).as_posix(),
            "bytes": vector_output.stat().st_size,
            "sha256": sha256_file(vector_output),
            "shape": list(mean_deltas.shape),
            "dtype": str(mean_deltas.dtype),
        },
        "tracked_outputs": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in repo_outputs[1:]
        ],
    }
    atomic_json(STATUS, status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
