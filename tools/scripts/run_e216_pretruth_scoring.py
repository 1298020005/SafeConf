#!/usr/bin/env python3
"""Freeze E216 risk scores and family certificates before reading test truth."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_MEMBERS = (
    ("latent", 1, 0.125),
    ("latent", 2, 0.125),
    ("latent", 3, 0.125),
    ("latent", 4, 0.125),
    ("linear", 1, 0.5),
)
TASK_KEYS = ("condition", "cell_type", "treatment")


class PretruthFailure(RuntimeError):
    """The frozen E216 pretruth score contract failed."""


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise PretruthFailure(f"invalid status file: {path}") from exc


def task_keys_from_npz(payload) -> list[tuple[str, str, str]]:
    return list(
        zip(
            payload["condition"].astype(str).tolist(),
            payload["cell_type"].astype(str).tolist(),
            payload["treatment"].astype(str).tolist(),
            strict=True,
        )
    )


def architecture_balanced_family(member_predictions: np.ndarray) -> dict[str, np.ndarray]:
    """Combine four latent seeds and one linear model with 0.5/0.5 architecture weight."""

    values = np.asarray(member_predictions, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] != 5 or not np.isfinite(values).all():
        raise PretruthFailure("family requires five finite member prediction matrices")
    weights = np.asarray([item[2] for item in EXPECTED_MEMBERS], dtype=np.float64)
    centroid = np.einsum("m,mtg->tg", weights, values)
    squared_distance = np.mean(np.square(values - centroid[None, :, :]), axis=2)
    lower_bound = np.sqrt(np.einsum("m,mt->t", weights, squared_distance))
    latent_centroid = values[:4].mean(axis=0)
    latent_disagreement = np.sqrt(
        np.mean(np.square(values[:4] - latent_centroid[None, :, :]), axis=(0, 2))
    )
    return {
        "weights": weights,
        "centroid": centroid,
        "lower_bound": lower_bound,
        "latent_centroid": latent_centroid,
        "latent_disagreement": latent_disagreement,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--prediction-queue-status", type=Path, required=True)
    parser.add_argument("--source-evidence", type=Path, required=True)
    parser.add_argument("--source-status", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / "code" / "safeconf_audit"))
    from safeconf_audit.e208_score import COMPONENT_NAMES, score_state_batch, task_features

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise PretruthFailure(f"refusing to overwrite: {output_dir}")
    queue_status = load_json(args.prediction_queue_status.resolve())
    if (
        queue_status.get("status") != "PREDICTIONS_COMPLETE"
        or queue_status.get("test_perturbed_expression_rows_read") != 0
    ):
        raise PretruthFailure("truth-free prediction queue is not complete")
    source_status = load_json(args.source_status.resolve())
    if (
        source_status.get("status") != "PASS"
        or source_status.get("test_perturbed_expression_rows_read") != 0
    ):
        raise PretruthFailure("train-only source evidence has not passed")

    tasks = pd.read_csv(args.tasks.resolve())
    tasks = tasks.sort_values(list(TASK_KEYS), kind="stable").reset_index(drop=True)
    expected_keys = list(tasks.loc[:, TASK_KEYS].itertuples(index=False, name=None))
    if len(expected_keys) != 224 or len(set(expected_keys)) != 224:
        raise PretruthFailure("registered task identity changed")

    prediction_root = args.predictions_root.resolve()
    member_arrays = []
    member_files = []
    control_payload = None
    gene_names = None
    for architecture, seed, _weight in EXPECTED_MEMBERS:
        member_dir = prediction_root / architecture / f"seed_{seed}"
        status_path = member_dir / "E216_PREDICTION_STATUS.json"
        status = load_json(status_path)
        if (
            status.get("status") != "COMPLETE"
            or status.get("task_count") != 224
            or status.get("test_perturbed_expression_rows_read") != 0
        ):
            raise PretruthFailure(f"prediction member is not sealed truth-free: {member_dir}")
        prediction_path = member_dir / "E216_TASK_CENTROID_PREDICTIONS.npz"
        current = np.load(prediction_path, allow_pickle=False)
        if task_keys_from_npz(current) != expected_keys:
            raise PretruthFailure(f"task order changed for {architecture}/seed_{seed}")
        current_genes = current["gene_names"].astype(str)
        if gene_names is None:
            gene_names = current_genes
        elif not np.array_equal(gene_names, current_genes):
            raise PretruthFailure("member gene axes differ")
        predictions = np.asarray(current["predictions"], dtype=np.float64)
        if predictions.shape != (224, 4000) or not np.isfinite(predictions).all():
            raise PretruthFailure("member prediction matrix is invalid")
        member_arrays.append(predictions)
        member_files.extend([status_path, prediction_path])
        control_path = member_dir / "E216_CONTROL_CENTROIDS.npz"
        control = np.load(control_path, allow_pickle=False)
        if control_payload is None:
            control_payload = {
                name: np.asarray(control[name])
                for name in ("control_centroids", "gene_names", "cell_type", "treatment", "n_controls")
            }
        else:
            for name, reference in control_payload.items():
                if not np.array_equal(reference, np.asarray(control[name])):
                    raise PretruthFailure("members did not use identical control centroids")
        member_files.append(control_path)

    members = np.stack(member_arrays)
    family = architecture_balanced_family(members)
    controls = {
        (str(cell), str(treatment)): np.asarray(vector, dtype=np.float64)
        for cell, treatment, vector in zip(
            control_payload["cell_type"],
            control_payload["treatment"],
            control_payload["control_centroids"],
            strict=True,
        )
    }
    if not np.array_equal(gene_names, control_payload["gene_names"].astype(str)):
        raise PretruthFailure("control and prediction gene axes differ")

    source = np.load(args.source_evidence.resolve(), allow_pickle=False)
    if not np.array_equal(gene_names, source["gene_names"].astype(str)):
        raise PretruthFailure("source and prediction gene axes differ")
    source_meta = pd.DataFrame(
        {
            "condition": source["condition"].astype(str),
            "cell_type": source["cell_type"].astype(str),
            "treatment": source["treatment"].astype(str),
            "n_source_cells": source["n_source_cells"].astype(int),
            "source_row": np.arange(len(source["condition"]), dtype=int),
        }
    )
    source_deltas = np.asarray(source["source_deltas"], dtype=np.float64)

    feature_rows = []
    latent_predictions = members[:4]
    for task_index, task in tasks.iterrows():
        key = (str(task.cell_type), str(task.treatment))
        if key not in controls:
            raise PretruthFailure(f"missing target control centroid: {key}")
        block = source_meta.loc[source_meta.condition.eq(str(task.condition))]
        block = block.loc[
            ~(block.cell_type.eq(key[0]) & block.treatment.eq(key[1]))
        ]
        features = task_features(
            latent_predictions[:, task_index, :],
            controls[key],
            source_deltas[block.source_row.to_numpy(dtype=int)],
            int(block.n_source_cells.sum()),
            total_other_contexts=29,
        )
        row = {
            **{name: str(task[name]) for name in TASK_KEYS},
            **{name: value for name, value in zip(COMPONENT_NAMES, features.components(), strict=True)},
            "predicted_magnitude": features.predicted_magnitude,
            "architecture_balanced_lower_bound": family["lower_bound"][task_index],
            "latent_family_disagreement": family["latent_disagreement"][task_index],
            "n_source_contexts": int(len(block)),
            "n_source_cells": int(block.n_source_cells.sum()),
        }
        feature_rows.append(row)
    scores = pd.DataFrame(feature_rows)
    scores["safeconf"] = np.nan
    scores["magnitude_percentile_rank"] = np.nan
    scores["safeconf_percentile_rank"] = np.nan
    scores["safeconf_m_4to1"] = np.nan
    state_rows = []
    for state, index in scores.groupby(["cell_type", "treatment"], sort=True).groups.items():
        take = np.asarray(sorted(index), dtype=int)
        result = score_state_batch(
            scores.loc[take, list(COMPONENT_NAMES)].to_numpy(dtype=float),
            scores.loc[take, "predicted_magnitude"].to_numpy(dtype=float),
        )
        for name in (
            "safeconf",
            "magnitude_percentile_rank",
            "safeconf_percentile_rank",
            "safeconf_m_4to1",
        ):
            scores.loc[take, name] = result[name]
        state_rows.append(
            {
                "cell_type": str(state[0]),
                "treatment": str(state[1]),
                "n_tasks": int(len(take)),
                **{
                    f"center_{name}": float(value)
                    for name, value in zip(COMPONENT_NAMES, result["component_center"], strict=True)
                },
                **{
                    f"scale_{name}": float(value)
                    for name, value in zip(COMPONENT_NAMES, result["component_scale"], strict=True)
                },
            }
        )
    if scores[["safeconf", "safeconf_m_4to1"]].isna().any().any():
        raise PretruthFailure("one or more registered states abstained")

    thresholds = pd.DataFrame(
        {
            "quantile": np.arange(0.1, 1.0, 0.1),
            "tau": np.quantile(family["lower_bound"], np.arange(0.1, 1.0, 0.1)),
        }
    )
    output_dir.mkdir(parents=True)
    score_path = output_dir / "E216_PRETRUTH_TASK_SCORES.csv"
    state_path = output_dir / "E216_PRETRUTH_STATE_STANDARDIZATION.csv"
    threshold_path = output_dir / "E216_CERTIFICATE_THRESHOLDS.csv"
    family_path = output_dir / "E216_ARCHITECTURE_BALANCED_FAMILY.npz"
    atomic_text(score_path, scores.to_csv(index=False))
    atomic_text(state_path, pd.DataFrame(state_rows).to_csv(index=False))
    atomic_text(threshold_path, thresholds.to_csv(index=False))
    np.savez_compressed(
        family_path,
        member_predictions=members.astype(np.float32),
        member_architecture=np.asarray([item[0] for item in EXPECTED_MEMBERS], dtype=str),
        member_seed=np.asarray([item[1] for item in EXPECTED_MEMBERS], dtype=np.int32),
        member_weight=family["weights"],
        family_centroid=family["centroid"].astype(np.float32),
        family_lower_bound=family["lower_bound"],
        gene_names=np.asarray(gene_names, dtype=str),
    )

    input_files = [
        args.tasks.resolve(),
        args.prediction_queue_status.resolve(),
        args.source_evidence.resolve(),
        args.source_status.resolve(),
        *member_files,
    ]
    output_files = [score_path, state_path, threshold_path, family_path]
    manifest = pd.DataFrame(
        [
            {"role": "input", "path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in input_files
        ]
        + [
            {"role": "output", "path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in output_files
        ]
    )
    manifest_path = output_dir / "E216_PRETRUTH_MANIFEST.csv"
    atomic_text(manifest_path, manifest.to_csv(index=False))
    status = {
        "experiment": "E216_jiang24_resource_bounded_confirmation",
        "stage": "D3_PRETRUTH_SCORE_FREEZE",
        "status": "PRETRUTH_COMPLETE_UNPUSHED",
        "created_at": now_iso(),
        "registered_tasks": 224,
        "registered_states": 12,
        "members": [f"{architecture}/seed_{seed}" for architecture, seed, _ in EXPECTED_MEMBERS],
        "architecture_weights": {"latent_total": 0.5, "linear_total": 0.5},
        "certificate_thresholds": 9,
        "all_states_evaluable": True,
        "test_truth_access": "NOT_AUTHORIZED",
        "test_perturbed_expression_rows_read": 0,
        "target_truth_used_for_selection": False,
        "manifest_sha256": sha256(manifest_path),
    }
    atomic_text(
        output_dir / "E216_PRETRUTH_STATUS.json",
        json.dumps(status, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
