#!/usr/bin/env python3
"""Seal E201 multi-seed risk features without reading target outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
OUT = ROOT / "docs/实验结果/E201_txpert_multitarget_retraining_20260802"
FREEZE = OUT / "PRETRUTH_TASK_BASE_FREEZE.md"
TASK_STATUS = OUT / "E201_PRETRUTH_TASK_BASE_STATUS.json"
TASKS = OUT / "tables/E201_PRETRUTH_TASK_BASE.csv"
RISK_TABLE = OUT / "tables/E201_PRETRUTH_RISK_FEATURES.csv"
RISK_STATUS = OUT / "E201_PRETRUTH_RISK_STATUS.json"
TARGETS = ("K562", "RPE1", "hepg2", "jurkat")
SEEDS = (1, 2, 3, 4)
EXPECTED_SAMPLES = {
    "K562": 150_472,
    "RPE1": 67_034,
    "hepg2": 54_911,
    "jurkat": 81_791,
}
RISK_COMPONENTS = (
    "family_disagreement",
    "model_source_gap",
    "source_delta_dispersion",
    "negative_log_source_cells",
    "support_context_deficit",
)


class RiskFailure(RuntimeError):
    pass


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--family-seal", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--vector-output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


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


def verify_git_release(family_seal: Path) -> str:
    required = (SCRIPT, FREEZE, TASK_STATUS, TASKS, family_seal)
    if not all(tracked_clean(path) for path in required):
        raise RiskFailure(
            "risk code/freeze/task base/family seal is not tracked and clean"
        )
    branch = git_text("branch", "--show-current")
    head = git_text("rev-parse", "HEAD")
    if not branch:
        raise RiskFailure("detached HEAD is not allowed")
    for remote in ("origin", "github"):
        if git_text("rev-parse", f"{remote}/{branch}") != head:
            raise RiskFailure(f"{remote}/{branch} differs from local HEAD")
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
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(temporary, path)


def rmse(left: np.ndarray, right: np.ndarray) -> float:
    difference = np.asarray(left, dtype=np.float64) - np.asarray(
        right, dtype=np.float64
    )
    return float(np.sqrt(np.mean(np.square(difference))))


def condition_from_label(target: str, label: str) -> str:
    prefix = f"{target}_"
    suffix = "_1+1"
    if not label.startswith(prefix) or not label.endswith(suffix):
        raise RiskFailure(f"unexpected target condition label: {label}")
    condition = label[len(prefix) : -len(suffix)]
    if not condition.endswith("+ctrl"):
        raise RiskFailure(f"not a single-perturbation label: {label}")
    return condition


def verify_task_base(data_root: Path) -> tuple[pd.DataFrame, np.ndarray, dict]:
    status = json.loads(TASK_STATUS.read_text(encoding="utf-8"))
    if (
        status.get("status") != "PASS"
        or int(status.get("n_tasks", -1)) != 2_008
        or int(status.get("n_primary_tasks", -1)) != 1_808
        or int(status.get("target_perturbed_expression_rows_opened", -1)) != 0
        or status.get("target_predictions_opened") is not False
        or status.get("target_outcomes_evaluated") is not False
    ):
        raise RiskFailure("pretruth task-base status failed")
    for entry in status.get("tracked_outputs", []):
        path = ROOT / entry["path"]
        if (
            not path.is_file()
            or path.stat().st_size != int(entry["bytes"])
            or sha256_file(path) != entry["sha256"]
        ):
            raise RiskFailure(f"tracked task-base output changed: {path}")
    vector_record = status["source_mean_delta_file"]
    vector_path_text = vector_record["path"]
    if not vector_path_text.startswith("DATA/"):
        raise RiskFailure("source-mean vector is not DATA-relative")
    vector_path = data_root / vector_path_text[len("DATA/") :]
    if (
        not vector_path.is_file()
        or vector_path.stat().st_size != int(vector_record["bytes"])
        or sha256_file(vector_path) != vector_record["sha256"]
    ):
        raise RiskFailure("source-mean vector changed")
    tasks = pd.read_csv(TASKS, keep_default_na=True)
    source_mean = np.load(vector_path, mmap_mode="r")
    if (
        len(tasks) != 2_008
        or tasks.task_id.nunique() != 2_008
        or source_mean.shape != (2_008, 3_352)
        or not np.array_equal(
            tasks.source_mean_delta_row.to_numpy(), np.arange(len(tasks))
        )
    ):
        raise RiskFailure("task base alignment changed")
    return tasks, source_mean, status


def verify_family_seal(path: Path) -> tuple[dict, str]:
    seal_sha = sha256_file(path)
    seal = json.loads(path.read_text(encoding="utf-8"))
    records = seal.get("records", [])
    jobs = {(record.get("target"), int(record.get("seed", -1))) for record in records}
    expected = {(target, seed) for target in TARGETS for seed in SEEDS}
    if (
        seal.get("status") != "SEALED_16_CHECKPOINTS"
        or int(seal.get("n_jobs", -1)) != 16
        or seal.get("target_truth_opened") is not False
        or jobs != expected
        or canonical_hash(records) != seal.get("records_sha256")
    ):
        raise RiskFailure("checkpoint family seal failed")
    return seal, seal_sha


def load_target_predictions(
    target: str,
    prediction_root: Path,
    family_seal_sha: str,
) -> tuple[pd.DataFrame, np.ndarray, list[np.ndarray], list[dict]]:
    target_root = prediction_root / target
    shared = target_root / "shared"
    shared_manifest_path = shared / "E201_SHARED_TARGET_MANIFEST.json"
    if not shared_manifest_path.is_file():
        raise RiskFailure(f"missing shared pretruth inputs: {target}")
    shared_manifest = json.loads(shared_manifest_path.read_text(encoding="utf-8"))
    if (
        shared_manifest.get("status") != "SEALED_PRETRUTH_TARGET_INPUTS"
        or shared_manifest.get("target") != target
        or int(shared_manifest.get("n_samples", -1)) != EXPECTED_SAMPLES[target]
        or int(shared_manifest.get("n_genes", -1)) != 3_352
        or shared_manifest.get("target_truth_materialized") is not False
        or int(shared_manifest.get("target_expression_nonzero_values_seen", -1)) != 0
    ):
        raise RiskFailure(f"shared pretruth manifest failed: {target}")
    by_role = {entry["role"]: entry for entry in shared_manifest.get("files", [])}
    if set(by_role) != {"controls", "observations"}:
        raise RiskFailure(f"shared file roles changed: {target}")
    for role, filename in (
        ("controls", "controls.npy"),
        ("observations", "observations.csv"),
    ):
        entry = by_role[role]
        path = shared / filename
        if (
            entry.get("path") != filename
            or not path.is_file()
            or path.stat().st_size != int(entry["bytes"])
            or sha256_file(path) != entry["sha256"]
        ):
            raise RiskFailure(f"shared {role} changed: {target}")
    observations = pd.read_csv(shared / "observations.csv", keep_default_na=False)
    controls = np.load(shared / "controls.npy", mmap_mode="r")
    if (
        len(observations) != EXPECTED_SAMPLES[target]
        or observations.row_index.tolist() != list(range(len(observations)))
        or set(observations.cell_type.astype(str)) != {target}
        or controls.shape != (EXPECTED_SAMPLES[target], 3_352)
    ):
        raise RiskFailure(f"shared target alignment changed: {target}")

    predictions = []
    statuses = []
    shared_manifest_sha = sha256_file(shared_manifest_path)
    for seed in SEEDS:
        run_dir = target_root / f"seed_{seed}"
        status_path = run_dir / "E201_PREDICTION_RUN.json"
        if not status_path.is_file():
            raise RiskFailure(f"missing prediction status: {target}/seed_{seed}")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if (
            status.get("status") != "COMPLETE"
            or status.get("target") != target
            or int(status.get("seed", -1)) != seed
            or status.get("checkpoint_role") != "last"
            or status.get("family_seal_sha256") != family_seal_sha
            or status.get("target_truth_materialized") is not False
            or int(status.get("target_expression_nonzero_values_seen", -1)) != 0
            or status.get("shared_target_manifest_sha256") != shared_manifest_sha
            or status.get("metadata_stream_sha256")
            != shared_manifest.get("metadata_stream_sha256")
            or status.get("control_stream_sha256")
            != shared_manifest.get("control_stream_sha256")
        ):
            raise RiskFailure(f"prediction status failed: {target}/seed_{seed}")
        prediction_record = status["prediction_file"]
        prediction_path = run_dir / prediction_record["path"]
        if (
            not prediction_path.is_file()
            or prediction_path.stat().st_size != int(prediction_record["bytes"])
            or sha256_file(prediction_path) != prediction_record["sha256"]
        ):
            raise RiskFailure(f"prediction file changed: {target}/seed_{seed}")
        prediction = np.load(prediction_path, mmap_mode="r")
        if prediction.shape != (EXPECTED_SAMPLES[target], 3_352):
            raise RiskFailure(f"prediction shape changed: {target}/seed_{seed}")
        predictions.append(prediction)
        statuses.append(status)
    return observations, controls, predictions, statuses


def standardize_with_primary(frame: pd.DataFrame, column: str) -> pd.Series:
    primary = frame.loc[frame.analysis_stratum.eq("primary_ge30"), column].astype(float)
    center = float(primary.mean())
    scale = float(primary.std(ddof=0))
    if not math.isfinite(center) or not math.isfinite(scale) or scale <= 0:
        raise RiskFailure(f"invalid within-target standardization: {column}")
    return (frame[column].astype(float) - center) / scale


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    family_seal_path = args.family_seal.resolve()
    prediction_root = args.prediction_root.resolve()
    vector_output_dir = args.vector_output_dir.resolve()
    if RISK_TABLE.exists() or RISK_STATUS.exists() or vector_output_dir.exists():
        raise RiskFailure("pretruth risk output already exists")
    forbidden = [
        path for path in prediction_root.rglob("*") if "truth" in path.name.lower()
    ]
    if forbidden:
        raise RiskFailure(
            f"target truth artifact exists before risk seal: {forbidden[:3]}"
        )
    safeconf_commit = verify_git_release(family_seal_path)
    tasks, source_mean, task_status = verify_task_base(data_root)
    _, family_seal_sha = verify_family_seal(family_seal_path)

    feature_blocks = []
    seed_centroid_blocks = []
    family_centroid_blocks = []
    control_centroid_blocks = []
    source_prediction_blocks = []
    prediction_status_records = []
    for target in TARGETS:
        target_tasks = tasks.loc[tasks.target.eq(target)].copy()
        observations, controls, predictions, statuses = load_target_predictions(
            target, prediction_root, family_seal_sha
        )
        prediction_status_records.extend(
            {
                "target": target,
                "seed": int(status["seed"]),
                "prediction_sha256": status["prediction_file"]["sha256"],
                "prediction_status_sha256": sha256_file(
                    prediction_root
                    / target
                    / f"seed_{int(status['seed'])}"
                    / "E201_PREDICTION_RUN.json"
                ),
            }
            for status in statuses
        )
        conditions = observations.pert_cond_name.astype(str).map(
            lambda label: condition_from_label(target, label)
        )
        condition_array = conditions.to_numpy()
        target_seed_centroids = []
        target_family_centroids = []
        target_control_centroids = []
        target_source_predictions = []
        rows = []
        for task in target_tasks.itertuples(index=False):
            indices = np.flatnonzero(condition_array == str(task.condition))
            if len(indices) != int(task.n_target_cells):
                raise RiskFailure(f"target cell count changed: {task.task_id}")
            seed_centroids = np.stack(
                [
                    np.asarray(prediction[indices], dtype=np.float64).mean(axis=0)
                    for prediction in predictions
                ]
            )
            if not np.isfinite(seed_centroids).all():
                raise RiskFailure(f"non-finite prediction centroid: {task.task_id}")
            family_centroid = seed_centroids.mean(axis=0)
            control_centroid = np.asarray(controls[indices], dtype=np.float64).mean(
                axis=0
            )
            source_delta = np.asarray(
                source_mean[int(task.source_mean_delta_row)], dtype=np.float64
            )
            source_prediction = control_centroid + source_delta
            disagreement = float(
                np.sqrt(np.mean(np.square(seed_centroids - family_centroid[None, :])))
            )
            radius = float(
                np.max(
                    np.sqrt(
                        np.mean(
                            np.square(seed_centroids - family_centroid[None, :]),
                            axis=1,
                        )
                    )
                )
            )
            rows.append(
                {
                    "task_id": task.task_id,
                    "family_disagreement": disagreement,
                    "family_radius": radius,
                    "predicted_magnitude": rmse(family_centroid, control_centroid),
                    "model_source_gap": rmse(family_centroid, source_prediction),
                    "source_transfer_magnitude": rmse(
                        source_prediction, control_centroid
                    ),
                }
            )
            target_seed_centroids.append(seed_centroids.astype(np.float32))
            target_family_centroids.append(family_centroid.astype(np.float32))
            target_control_centroids.append(control_centroid.astype(np.float32))
            target_source_predictions.append(source_prediction.astype(np.float32))
        raw_features = pd.DataFrame(rows)
        target_features = target_tasks.merge(
            raw_features, on="task_id", how="left", validate="one_to_one"
        )
        for component in RISK_COMPONENTS:
            target_features[f"z_{component}"] = standardize_with_primary(
                target_features, component
            )
        target_features["safeconf_e201_risk"] = target_features[
            [f"z_{component}" for component in RISK_COMPONENTS]
        ].mean(axis=1)
        feature_blocks.append(target_features)
        seed_centroid_blocks.append(np.stack(target_seed_centroids, axis=1))
        family_centroid_blocks.append(np.stack(target_family_centroids))
        control_centroid_blocks.append(np.stack(target_control_centroids))
        source_prediction_blocks.append(np.stack(target_source_predictions))

    features = pd.concat(feature_blocks, ignore_index=True)
    seed_centroids = np.concatenate(seed_centroid_blocks, axis=1)
    family_centroids = np.concatenate(family_centroid_blocks, axis=0)
    control_centroids = np.concatenate(control_centroid_blocks, axis=0)
    source_predictions = np.concatenate(source_prediction_blocks, axis=0)
    arrays = {
        "E201_SEED_CENTROIDS.npy": seed_centroids,
        "E201_FAMILY_CENTROIDS.npy": family_centroids,
        "E201_CONTROL_CENTROIDS.npy": control_centroids,
        "E201_SOURCE_TRANSFER_CENTROIDS.npy": source_predictions,
    }
    if (
        len(features) != 2_008
        or features.task_id.nunique() != 2_008
        or seed_centroids.shape != (4, 2_008, 3_352)
        or family_centroids.shape != (2_008, 3_352)
        or control_centroids.shape != (2_008, 3_352)
        or source_predictions.shape != (2_008, 3_352)
        or not all(np.isfinite(values).all() for values in arrays.values())
        or not np.allclose(seed_centroids.mean(axis=0), family_centroids, atol=1e-6)
    ):
        raise RiskFailure("combined pretruth feature contract failed")
    risk_identity = np.max(
        np.abs(
            features.safeconf_e201_risk.to_numpy()
            - features[[f"z_{component}" for component in RISK_COMPONENTS]].mean(axis=1)
        )
    )
    if risk_identity > 1e-12:
        raise RiskFailure("SafeConf risk identity changed")

    vector_output_dir.mkdir(parents=True)
    vector_records = []
    for filename, values in arrays.items():
        path = vector_output_dir / filename
        atomic_npy(path, values.astype(np.float32, copy=False))
        vector_records.append(
            {
                "path": "DATA/" + path.relative_to(data_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "shape": list(values.shape),
                "dtype": "float32",
            }
        )
    atomic_csv(RISK_TABLE, features)
    status = {
        "experiment": "E201_txpert_multitarget_retraining",
        "stage": "PRETRUTH_RISK_FEATURES",
        "status": "PASS",
        "generated_at": now(),
        "safeconf_commit": safeconf_commit,
        "family_seal_sha256": family_seal_sha,
        "task_base_status_sha256": sha256_file(TASK_STATUS),
        "n_tasks": len(features),
        "n_primary_tasks": int(features.analysis_stratum.eq("primary_ge30").sum()),
        "risk_components": list(RISK_COMPONENTS),
        "standardization": "within target; parameters from primary_ge30 tasks",
        "risk_identity_max_abs_residual": float(risk_identity),
        "target_expression_nonzero_values_seen": 0,
        "target_truth_materialized": False,
        "target_outcomes_evaluated": False,
        "risk_table": {
            "path": RISK_TABLE.relative_to(ROOT).as_posix(),
            "bytes": RISK_TABLE.stat().st_size,
            "sha256": sha256_file(RISK_TABLE),
        },
        "vector_files": vector_records,
        "prediction_inputs": prediction_status_records,
    }
    atomic_json(RISK_STATUS, status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
