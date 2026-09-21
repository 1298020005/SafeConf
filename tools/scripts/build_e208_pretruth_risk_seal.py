#!/usr/bin/env python3
"""Build the complete E208 risk and certificate table before test-truth access."""

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


JOBS = (("latent", 1), ("latent", 2), ("latent", 3), ("latent", 4), ("linear", 1))


class RiskSealFailure(RuntimeError):
    """A frozen pretruth input or risk calculation failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, value: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    value.to_csv(temporary, index=False)
    os.replace(temporary, path)


def load_prediction(root: Path, stage: str, architecture: str, seed: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
    directory = root / stage / architecture / f"seed_{seed}"
    if stage == "validation":
        prefix = "E208_VALIDATION_"
        status_name = "E208_VALIDATION_PREDICTION_STATUS.json"
        expected = 216
    else:
        prefix = "E208_"
        status_name = "E208_PREDICTION_STATUS.json"
        expected = 224
    prediction_path = directory / f"{prefix}PREDICTION_CENTROIDS.npy"
    control_path = directory / f"{prefix}CONTROL_CENTROIDS.npy"
    tasks_path = directory / f"{prefix}PREDICTION_TASKS.csv"
    if stage == "validation":
        tasks_path = directory / "E208_VALIDATION_TASKS.csv"
    status_path = directory / status_name
    for path in (prediction_path, control_path, tasks_path, status_path):
        if not path.is_file():
            raise RiskSealFailure(f"prediction artifact missing: {path}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if (
        status.get("status") != "PASS"
        or status.get("architecture") != architecture
        or int(status.get("seed", -1)) != seed
        or int(status.get("n_tasks", -1)) != expected
        or int(status.get("test_perturbed_expression_rows_read", -1)) != 0
    ):
        raise RiskSealFailure(f"prediction status failed: {directory}")
    predictions = np.load(prediction_path, allow_pickle=False)
    controls = np.load(control_path, allow_pickle=False)
    tasks = pd.read_csv(tasks_path)
    if predictions.shape != (expected, 15473) or controls.shape != predictions.shape:
        raise RiskSealFailure(f"prediction shape changed: {directory}")
    return predictions, controls, tasks, status


def family_geometry(members: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    latent_mean = sum(array.astype(np.float64) for array in members[:4]) / 4.0
    linear = members[4].astype(np.float64)
    family_mean = 0.5 * latent_mean + 0.5 * linear
    squared = np.zeros(len(family_mean), dtype=np.float64)
    for latent in members[:4]:
        squared += 0.125 * np.mean((latent.astype(np.float64) - family_mean) ** 2, axis=1)
    squared += 0.5 * np.mean((linear - family_mean) ** 2, axis=1)
    return family_mean, np.sqrt(squared)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--validation-cache-dir", type=Path, required=True)
    parser.add_argument("--competence-dir", type=Path, required=True)
    parser.add_argument("--source-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prediction_root = args.prediction_root.expanduser().absolute()
    validation_cache = args.validation_cache_dir.expanduser().absolute()
    competence_dir = args.competence_dir.expanduser().absolute()
    source_cache = args.source_cache_dir.expanduser().absolute()
    output = args.output_dir.expanduser().absolute()

    competence_path = competence_dir / "E208_VALIDATION_COMPETENCE_STATUS.json"
    competence = json.loads(competence_path.read_text(encoding="utf-8"))
    if competence.get("status") != "PASS" or competence.get("test_perturbed_expression_rows_read") != 0:
        raise RiskSealFailure("registered family did not pass validation competence")
    validation_status_path = validation_cache / "E208_VALIDATION_CACHE_STATUS.json"
    validation_status = json.loads(validation_status_path.read_text(encoding="utf-8"))
    truth_path = validation_cache / "E208_VALIDATION_TRUTH_CENTROIDS.npy"
    if sha256_file(truth_path) != validation_status["truth_centroids"]["sha256"]:
        raise RiskSealFailure("validation truth centroid checksum changed")
    validation_truth = np.load(truth_path, allow_pickle=False).astype(np.float64)

    source_status_path = source_cache / "E208_SOURCE_EFFECT_CACHE_STATUS.json"
    source_status = json.loads(source_status_path.read_text(encoding="utf-8"))
    source_effect_path = source_cache / "E208_SOURCE_EFFECTS.npy"
    source_manifest_path = source_cache / "E208_SOURCE_EFFECT_MANIFEST.csv"
    if (
        source_status.get("status") != "PASS"
        or source_status.get("test_perturbed_expression_rows_read") != 0
        or sha256_file(source_effect_path) != source_status["effects"]["sha256"]
        or sha256_file(source_manifest_path) != source_status["manifest"]["sha256"]
    ):
        raise RiskSealFailure("train-source cache gate failed")
    source_effects = np.load(source_effect_path, allow_pickle=False).astype(np.float64)
    source_manifest = pd.read_csv(source_manifest_path)
    if source_effects.shape != (238, 15473) or len(source_manifest) != 238:
        raise RiskSealFailure("train-source cache shape changed")

    validation_members = []
    validation_controls = None
    validation_tasks = None
    test_members = []
    test_controls = None
    test_tasks = None
    input_records = []
    for architecture, seed in JOBS:
        vp, vc, vt, vs = load_prediction(
            prediction_root, "validation", architecture, seed
        )
        tp, tc, tt, ts = load_prediction(
            prediction_root, "test_pretruth", architecture, seed
        )
        if validation_controls is None:
            validation_controls, validation_tasks = vc, vt
        elif not np.array_equal(vc, validation_controls) or not vt.equals(validation_tasks):
            raise RiskSealFailure("validation models used different task/control inputs")
        if test_controls is None:
            test_controls, test_tasks = tc, tt
        elif not np.array_equal(tc, test_controls) or not tt.equals(test_tasks):
            raise RiskSealFailure("test models used different task/control inputs")
        validation_members.append(vp)
        test_members.append(tp)
        input_records.append(
            {
                "architecture": architecture,
                "seed": seed,
                "validation_checkpoint_sha256": vs["checkpoint_sha256"],
                "validation_prediction_sha256": vs["prediction_centroids_sha256"],
                "test_checkpoint_sha256": ts["checkpoint_sha256"],
                "test_prediction_sha256": ts["prediction_centroids"]["sha256"],
            }
        )
    if test_tasks.task_id.nunique() != 224 or validation_tasks.task_id.nunique() != 216:
        raise RiskSealFailure("prediction task identity changed")

    _, validation_family_lower = family_geometry(validation_members)
    validation_family_error_sq = np.zeros(len(validation_truth), dtype=np.float64)
    weights = [0.125, 0.125, 0.125, 0.125, 0.5]
    for weight, member in zip(weights, validation_members, strict=True):
        validation_family_error_sq += weight * np.mean(
            (member.astype(np.float64) - validation_truth) ** 2, axis=1
        )
    validation_family_error = np.sqrt(validation_family_error_sq)
    calibration = validation_tasks[["task_id", "cell_type", "treatment", "condition"]].copy()
    calibration["registered_family_rmse"] = validation_family_error
    calibration["registered_family_lower_bound"] = validation_family_lower
    calibration_targets = (
        calibration.groupby("condition", sort=True)
        .registered_family_rmse.max()
        .rename("max_registered_family_rmse")
        .reset_index()
    )
    if len(calibration_targets) < 50:
        raise RiskSealFailure("fewer than 50 validation calibration genes")
    conformal_rank = min(
        len(calibration_targets), math.ceil((len(calibration_targets) + 1) * 0.9)
    )
    conformal_q90 = float(
        np.sort(calibration_targets.max_registered_family_rmse.to_numpy())[conformal_rank - 1]
    )

    test_family_mean, test_family_lower = family_geometry(test_members)
    latent_stack = np.stack([item.astype(np.float64) for item in test_members[:4]])
    latent_mean = latent_stack.mean(axis=0)
    same_arch_disagreement = np.sqrt(
        np.mean((latent_stack - latent_mean[None, :, :]) ** 2, axis=(0, 2))
    )
    x0 = test_controls.astype(np.float64)
    risk = test_tasks.copy()
    risk["predicted_magnitude"] = np.sqrt(np.mean((latent_mean - x0) ** 2, axis=1))
    risk["same_arch_seed_disagreement"] = same_arch_disagreement
    risk["registered_family_lower_bound"] = test_family_lower
    risk["registered_family_conformal_upper"] = np.sqrt(
        conformal_q90**2 + test_family_lower**2
    )

    raw_rows = []
    for task_index, task in enumerate(test_tasks.itertuples(index=False)):
        target_context = f"{task.cell_type}|{task.treatment}"
        sources = source_manifest.loc[
            source_manifest.condition.astype(str).eq(str(task.condition))
            & ~source_manifest.source_context_id.astype(str).eq(target_context)
        ].copy()
        if len(sources) < 2:
            raise RiskSealFailure(f"fewer than two train sources for {task.task_id}")
        effects = source_effects[sources.effect_row.to_numpy(np.int64)]
        mean_effect = effects.mean(axis=0)
        raw_rows.append(
            {
                "task_id": task.task_id,
                "D": same_arch_disagreement[task_index],
                "G": float(np.sqrt(np.mean((latent_mean[task_index] - (x0[task_index] + mean_effect)) ** 2))),
                "H": float(np.sqrt(np.mean((effects - mean_effect[None, :]) ** 2))),
                "N": float(-np.log1p(sources.n_perturbed_cells.sum())),
                "C": float(29 - len(sources)),
                "n_source_contexts": len(sources),
                "n_source_perturbed_cells": int(sources.n_perturbed_cells.sum()),
            }
        )
    risk = risk.merge(pd.DataFrame(raw_rows), on="task_id", how="left", validate="one_to_one")
    risk["risk_status"] = "PASS"
    z_columns = []
    for feature in ("D", "G", "H", "N", "C"):
        z_name = f"z_{feature}"
        z_columns.append(z_name)
        risk[z_name] = np.nan
    for (cell, treatment), index in risk.groupby(["cell_type", "treatment"], sort=True).groups.items():
        index = list(index)
        if len(index) < 10:
            risk.loc[index, "risk_status"] = "ABSTAIN_TOO_FEW_TASKS"
            continue
        failed = False
        for feature, z_name in zip(("D", "G", "H", "N", "C"), z_columns, strict=True):
            values = risk.loc[index, feature].to_numpy(np.float64)
            standard_deviation = values.std(ddof=0)
            if not np.isfinite(values).all() or not np.isfinite(standard_deviation) or standard_deviation == 0:
                failed = True
                break
            risk.loc[index, z_name] = (values - values.mean()) / standard_deviation
        if failed:
            risk.loc[index, "risk_status"] = "ABSTAIN_STANDARDIZATION"
    if not risk.risk_status.eq("PASS").all():
        failed_states = sorted(
            set(
                risk.loc[~risk.risk_status.eq("PASS"), ["cell_type", "treatment"]]
                .astype(str)
                .agg("|".join, axis=1)
            )
        )
        raise RiskSealFailure(f"registered risk score abstained: {failed_states}")
    risk["safeconf_original"] = risk[z_columns].mean(axis=1)
    risk["magnitude_rank"] = np.nan
    risk["safeconf_rank"] = np.nan
    risk["safeconf_m_primary"] = np.nan
    risk["safeconf_oneway_secondary"] = np.nan
    for _, index in risk.groupby(["cell_type", "treatment"], sort=True).groups.items():
        index = list(index)
        n = len(index)
        magnitude_rank = risk.loc[index, "predicted_magnitude"].rank(method="average", ascending=True)
        safeconf_rank = risk.loc[index, "safeconf_original"].rank(method="average", ascending=True)
        m = magnitude_rank / n
        s = safeconf_rank / n
        risk.loc[index, "magnitude_rank"] = magnitude_rank
        risk.loc[index, "safeconf_rank"] = safeconf_rank
        risk.loc[index, "safeconf_m_primary"] = (4.0 * magnitude_rank + safeconf_rank) / (5.0 * n)
        risk.loc[index, "safeconf_oneway_secondary"] = m + 0.25 * np.maximum(s - m, 0.0)
    if not np.isfinite(risk.select_dtypes(include=[np.number]).to_numpy()).all():
        raise RiskSealFailure("non-finite values entered formal risk table")

    quantiles = np.arange(0.1, 1.0, 0.1)
    threshold_table = pd.DataFrame(
        {
            "quantile": quantiles,
            "tau": np.quantile(test_family_lower, quantiles, method="linear"),
            "method": "numpy_linear",
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    risk_path = output / "E208_PRETRUTH_RISK_FEATURES.csv"
    calibration_task_path = output / "E208_VALIDATION_FAMILY_ERRORS.csv"
    calibration_target_path = output / "E208_VALIDATION_TARGET_MAX_ERRORS.csv"
    threshold_path = output / "E208_REGISTERED_FAMILY_THRESHOLDS.csv"
    atomic_csv(risk_path, risk)
    atomic_csv(calibration_task_path, calibration)
    atomic_csv(calibration_target_path, calibration_targets)
    atomic_csv(threshold_path, threshold_table)
    result = {
        "experiment": "E208_jiang24_external_confirmation",
        "stage": "D1_PRETRUTH_RISK_AND_CERTIFICATE_SEAL",
        "status": "PASS_AWAITING_REMOTE_SEAL",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "n_tasks": len(risk),
        "n_contexts": len(risk[["cell_type", "treatment"]].drop_duplicates()),
        "n_target_genes": int(risk.condition.nunique()),
        "n_validation_calibration_genes": len(calibration_targets),
        "conformal_rank_1_based": conformal_rank,
        "conformal_q90": conformal_q90,
        "family_weights": {
            "latent_seed_1": 0.125,
            "latent_seed_2": 0.125,
            "latent_seed_3": 0.125,
            "latent_seed_4": 0.125,
            "linear_seed_1": 0.5,
        },
        "risk_table": {"path": str(risk_path), "sha256": sha256_file(risk_path)},
        "calibration_tasks": {"path": str(calibration_task_path), "sha256": sha256_file(calibration_task_path)},
        "calibration_targets": {"path": str(calibration_target_path), "sha256": sha256_file(calibration_target_path)},
        "family_thresholds": {"path": str(threshold_path), "sha256": sha256_file(threshold_path)},
        "source_cache_status_sha256": sha256_file(source_status_path),
        "validation_cache_status_sha256": sha256_file(validation_status_path),
        "competence_status_sha256": sha256_file(competence_path),
        "prediction_inputs": input_records,
        "risk_formula": "(4*rank(predicted_magnitude)+rank(mean(zD,zG,zH,zN,zC)))/(5*n_context_tasks)",
        "test_perturbed_expression_rows_read": 0,
        "target_truth_access": "NOT_AUTHORIZED",
        "next_required_gate": "commit and push this seal to GitHub and Gitee before truth authorization",
    }
    atomic_json(output / "E208_PRETRUTH_RISK_SEAL_STATUS.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
