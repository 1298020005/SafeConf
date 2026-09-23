#!/usr/bin/env python3
"""Freeze E235 Jiang24 M+H and same-task baselines without test perturbation truth.

This is deliberately separate from E208's five-member latent/linear family:
E235 uses the four E233 corrected-linear seeds selected by the validation gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = ("D", "G", "H", "N", "C")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_file(path: Path, expected: str) -> None:
    if not path.is_file() or sha256(path) != expected:
        raise RuntimeError(f"missing or changed frozen input: {path}")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_predictions(root: Path, expected_tasks: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    members, reference_controls, records = [], None, []
    for seed in (1, 2, 3, 4):
        folder = root / f"seed_{seed}"
        status_path = folder / "E208_PREDICTION_STATUS.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if (status.get("status") != "PASS" or status.get("architecture") != "linear"
                or status.get("seed") != seed or status.get("n_tasks") != 224
                or status.get("test_perturbed_expression_rows_read") != 0):
            raise RuntimeError(f"seed {seed} did not satisfy the no-truth prediction contract")
        prediction_path = folder / "E208_PREDICTION_CENTROIDS.npy"
        controls_path = folder / "E208_CONTROL_CENTROIDS.npy"
        tasks_path = folder / "E208_PREDICTION_TASKS.csv"
        checked_file(prediction_path, status["prediction_centroids"]["sha256"])
        checked_file(controls_path, status["control_centroids"]["sha256"])
        checked_file(tasks_path, status["task_manifest"]["sha256"])
        tasks = pd.read_csv(tasks_path)
        if not tasks.equals(expected_tasks):
            raise RuntimeError(f"seed {seed} task identity/order differed from manifest")
        prediction = np.load(prediction_path, allow_pickle=False).astype(np.float64)
        controls = np.load(controls_path, allow_pickle=False).astype(np.float64)
        if prediction.shape != (224, 15473) or controls.shape != prediction.shape:
            raise RuntimeError(f"seed {seed} prediction/control shape mismatch")
        if not np.isfinite(prediction).all() or not np.isfinite(controls).all():
            raise RuntimeError(f"seed {seed} nonfinite prediction/control")
        if reference_controls is None:
            reference_controls = controls
        elif not np.array_equal(reference_controls, controls):
            raise RuntimeError("the four seeds used different target controls")
        members.append(prediction)
        records.append({"seed": seed, "status_sha256": sha256(status_path),
                        "prediction_sha256": sha256(prediction_path),
                        "checkpoint_sha256": status["checkpoint_sha256"]})
    return np.stack(members), reference_controls, records


def score_tasks(members: np.ndarray, controls: np.ndarray, tasks: pd.DataFrame,
                source_effects: np.ndarray, source_manifest: pd.DataFrame) -> pd.DataFrame:
    """Use only four model outputs, test controls, and training-source effects."""
    if members.shape != (4, 224, 15473) or controls.shape != (224, 15473):
        raise RuntimeError("E235 prediction tensor dimensions changed")
    mean_prediction = members.mean(axis=0)
    magnitude = np.sqrt(np.mean((mean_prediction - controls) ** 2, axis=1))
    disagreement = np.sqrt(np.mean((members - mean_prediction[None, :, :]) ** 2, axis=(0, 2)))
    rows = []
    for position, task in enumerate(tasks.itertuples(index=False)):
        target_context = f"{task.cell_type}|{task.treatment}"
        history = source_manifest.loc[
            source_manifest.condition.astype(str).eq(str(task.condition))
            & ~source_manifest.source_context_id.astype(str).eq(target_context)
        ]
        if len(history) < 2:
            raise RuntimeError(f"E235 ABSTAIN: fewer than 2 train sources for {task.task_id}")
        effects = source_effects[history.effect_row.to_numpy(np.int64)]
        mean_effect = effects.mean(axis=0)
        rows.append({
            "task_id": task.task_id, "cell_type": task.cell_type,
            "treatment": task.treatment, "condition": task.condition,
            "M": float(magnitude[position]),
            "D": float(disagreement[position]),
            "G": float(np.sqrt(np.mean((mean_prediction[position] -
                                        (controls[position] + mean_effect)) ** 2))),
            "H": float(np.sqrt(np.mean((effects - mean_effect[None, :]) ** 2))),
            "N": float(-np.log1p(history.n_perturbed_cells.sum())),
            "C": float(29 - len(history)),
            "n_source_contexts": len(history),
            "n_source_perturbed_cells": int(history.n_perturbed_cells.sum()),
            "source_context_ids": "|".join(sorted(history.source_context_id.astype(str))),
            "test_perturbed_expression_rows_read": 0,
        })
    frame = pd.DataFrame(rows)
    if len(frame) != 224 or frame.task_id.nunique() != 224:
        raise RuntimeError("E235 task list did not remain complete and unique")
    frame["score_random_fixed"] = frame.task_id.map(
        lambda task_id: int(hashlib.sha256(f"E235_RANDOM_V1\0{task_id}".encode()).hexdigest()[:16], 16)
        / 2**64
    )
    for _, indexes in frame.groupby(["cell_type", "treatment"], sort=True).groups.items():
        indexes = list(indexes)
        if len(indexes) < 10:
            raise RuntimeError("E235 state has fewer than ten test tasks")
        n = len(indexes)
        ranks = {}
        for feature in ("M", *FEATURES):
            values = frame.loc[indexes, feature].to_numpy(float)
            if not np.isfinite(values).all() or np.std(values) <= 0:
                raise RuntimeError(f"E235 nonfinite or constant {feature}")
            ranks[feature] = frame.loc[indexes, feature].rank(method="average") / n
            frame.loc[indexes, f"rank_{feature}"] = ranks[feature]
        for feature in FEATURES:
            frame.loc[indexes, f"score_M_plus_{feature}"] = .8 * ranks["M"] + .2 * ranks[feature]
            frame.loc[indexes, f"z_{feature}"] = ((frame.loc[indexes, feature] -
                                                    frame.loc[indexes, feature].mean()) /
                                                   frame.loc[indexes, feature].std(ddof=0))
        old_five = frame.loc[indexes, [f"z_{feature}" for feature in FEATURES]].mean(axis=1)
        frame.loc[indexes, "score_original_five_80_20"] = .8 * ranks["M"] + .2 * old_five.rank(method="average") / n
    if not np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all():
        raise RuntimeError("E235 score table contains nonfinite values")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stage1-status", "stage1-run-root", "stage2-status", "stage2-run-root",
                 "prediction-root", "task-manifest", "manifest-status", "source-cache-dir",
                 "control-cache-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    status_path = output / "E235_PRETRUTH_SCORE_STATUS.json"
    if status_path.exists() or (output / "E235_PRETRUTH_SCORES.csv").exists():
        raise RuntimeError("refusing to overwrite an E235 score seal")
    stage1 = json.loads(args.stage1_status.read_text(encoding="utf-8"))
    stage2 = json.loads(args.stage2_status.read_text(encoding="utf-8"))
    if (stage2.get("status") != "STAGE2_FAMILY_PASS"
            or stage2.get("test_perturbed_expression_rows_read") != 0
            or stage2.get("family_validation", {}).get("competence_gate") != "PASS"):
        raise RuntimeError("E233 four-seed family has not passed its validation competence gate")
    variant = stage2.get("selected_variant")
    if (stage1.get("status") != "STAGE1_PASS" or variant not in
            ("matched_control_linear", "matched_control_softplus")
            or stage1.get("selected_variant") != variant):
        raise RuntimeError("E235 selected predictor variant differs from E233 stage 1")
    manifest_status = json.loads(args.manifest_status.read_text(encoding="utf-8"))
    if manifest_status.get("status") != "PASS" or manifest_status.get("n_tasks") != 224:
        raise RuntimeError("E208 task manifest gate failed")
    checked_file(args.task_manifest, manifest_status["manifest_sha256"])
    tasks = pd.read_csv(args.task_manifest)
    if len(tasks) != 224 or tasks.task_id.nunique() != 224:
        raise RuntimeError("E235 task manifest incomplete")
    source_status_path = args.source_cache_dir / "E208_SOURCE_EFFECT_CACHE_STATUS.json"
    source_status = json.loads(source_status_path.read_text(encoding="utf-8"))
    if source_status.get("status") != "PASS" or source_status.get("test_perturbed_expression_rows_read") != 0:
        raise RuntimeError("train-only source cache gate failed")
    source_effect_path = args.source_cache_dir / "E208_SOURCE_EFFECTS.npy"
    source_manifest_path = args.source_cache_dir / "E208_SOURCE_EFFECT_MANIFEST.csv"
    gene_axis_path = args.source_cache_dir / "E208_SOURCE_GENE_AXIS.csv"
    checked_file(source_effect_path, source_status["effects"]["sha256"])
    checked_file(source_manifest_path, source_status["manifest"]["sha256"])
    checked_file(gene_axis_path, source_status["gene_axis"]["sha256"])
    control_status_path = args.control_cache_dir / "E208_CONTROL_CACHE_STATUS.json"
    control_status = json.loads(control_status_path.read_text(encoding="utf-8"))
    if control_status.get("status") != "PASS" or control_status.get("test_perturbed_expression_rows_read") != 0:
        raise RuntimeError("target control-only cache gate failed")
    checked_file(args.control_cache_dir / "E208_TARGET_CONTROL_CACHE.npz", control_status["cache"]["sha256"])
    with np.load(args.control_cache_dir / "E208_TARGET_CONTROL_CACHE.npz", allow_pickle=False) as packed:
        control_genes = packed["gene_names"].astype(str)
    source_genes = pd.read_csv(gene_axis_path).gene_name.astype(str).to_numpy()
    if not np.array_equal(control_genes, source_genes):
        raise RuntimeError("history and prediction gene axes differ")
    source_effects = np.load(source_effect_path, allow_pickle=False).astype(np.float64)
    source_manifest = pd.read_csv(source_manifest_path)
    if source_effects.shape != (238, 15473) or len(source_manifest) != 238:
        raise RuntimeError("E208 train-source data contract changed")
    members, controls, inputs = load_predictions(args.prediction_root, tasks)
    for record in inputs:
        seed = record["seed"]
        run = ((args.stage1_run_root / variant / str(stage1["attempt"])) if seed == 1
               else (args.stage2_run_root / variant / f"seed_{seed}"))
        training = json.loads((run / "E233_RUN_STATUS.json").read_text(encoding="utf-8"))
        if (training.get("status") != "COMPLETE" or training.get("variant") != variant
                or training.get("seed") != seed or
                record["checkpoint_sha256"] not in {item["sha256"] for item in training["checkpoints"]}):
            raise RuntimeError(f"seed {seed} prediction did not come from the selected E233 training run")
        record["training_status_sha256"] = sha256(run / "E233_RUN_STATUS.json")
    scores = score_tasks(members, controls, tasks, source_effects, source_manifest)
    score_path = output / "E235_PRETRUTH_SCORES.csv"
    write_csv(score_path, scores)
    record = {
        "status": "SCORES_READY_AWAITING_REMOTE_SEAL", "created_at": datetime.now().astimezone().isoformat(),
        "n_tasks": len(scores), "n_states": len(scores[["cell_type", "treatment"]].drop_duplicates()),
        "n_target_genes": scores.condition.nunique(), "score_sha256": sha256(score_path),
        "task_manifest_sha256": sha256(args.task_manifest),
        "source_cache_status_sha256": sha256(source_status_path),
        "control_cache_status_sha256": sha256(control_status_path),
        "stage1_status_sha256": sha256(args.stage1_status),
        "stage2_status_sha256": sha256(args.stage2_status), "selected_variant": variant,
        "prediction_inputs": inputs, "test_perturbed_expression_rows_read": 0,
        "target_truth_access": "NOT_AUTHORIZED",
        "registered_primary_score": "score_M_plus_H",
        "registered_comparator": "rank_M",
        "registered_formula": "0.8*rank_state(M)+0.2*rank_state(H)",
        "random_baseline": "SHA256(E235_RANDOM_V1\\0task_id)/2**64; frozen before truth",
    }
    write_json(status_path, record)
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
