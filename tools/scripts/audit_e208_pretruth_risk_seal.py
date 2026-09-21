#!/usr/bin/env python3
"""Independently reconstruct and audit the E208 pretruth risk seal."""

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
TOLERANCE = 1e-10


class AuditFailure(RuntimeError):
    """The sealed table differs from an independent reconstruction."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_prediction(root: Path, stage: str, architecture: str, seed: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
    directory = root / stage / architecture / f"seed_{seed}"
    if stage == "validation":
        prediction = directory / "E208_VALIDATION_PREDICTION_CENTROIDS.npy"
        control = directory / "E208_VALIDATION_CONTROL_CENTROIDS.npy"
        tasks = directory / "E208_VALIDATION_TASKS.csv"
        status_path = directory / "E208_VALIDATION_PREDICTION_STATUS.json"
        expected = 216
    else:
        prediction = directory / "E208_PREDICTION_CENTROIDS.npy"
        control = directory / "E208_CONTROL_CENTROIDS.npy"
        tasks = directory / "E208_PREDICTION_TASKS.csv"
        status_path = directory / "E208_PREDICTION_STATUS.json"
        expected = 224
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if (
        status.get("status") != "PASS"
        or status.get("architecture") != architecture
        or int(status.get("seed", -1)) != seed
        or int(status.get("n_tasks", -1)) != expected
        or int(status.get("test_perturbed_expression_rows_read", -1)) != 0
    ):
        raise AuditFailure(f"prediction status failed: {directory}")
    arrays = np.load(prediction, allow_pickle=False).astype(np.float64)
    controls = np.load(control, allow_pickle=False).astype(np.float64)
    task_frame = pd.read_csv(tasks)
    if arrays.shape != (expected, 15473) or controls.shape != arrays.shape or len(task_frame) != expected:
        raise AuditFailure(f"prediction shape failed: {directory}")
    return arrays, controls, task_frame, status


def family_geometry(members: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    latent_mean = sum(members[:4]) / 4.0
    family_mean = 0.5 * latent_mean + 0.5 * members[4]
    squared = sum(0.125 * np.mean((item - family_mean) ** 2, axis=1) for item in members[:4])
    squared += 0.5 * np.mean((members[4] - family_mean) ** 2, axis=1)
    return family_mean, np.sqrt(squared)


def max_abs(left: np.ndarray, right: np.ndarray, label: str) -> float:
    if left.shape != right.shape:
        raise AuditFailure(f"shape mismatch for {label}: {left.shape} != {right.shape}")
    difference = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
    if not np.isfinite(difference) or difference > TOLERANCE:
        raise AuditFailure(f"{label} maximum absolute difference {difference} exceeds {TOLERANCE}")
    return difference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--validation-cache-dir", type=Path, required=True)
    parser.add_argument("--source-cache-dir", type=Path, required=True)
    parser.add_argument("--seal-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for name in ("prediction_root", "validation_cache_dir", "source_cache_dir", "seal_dir", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().absolute())

    status_path = args.seal_dir / "E208_PRETRUTH_RISK_SEAL_STATUS.json"
    risk_path = args.seal_dir / "E208_PRETRUTH_RISK_FEATURES.csv"
    calibration_path = args.seal_dir / "E208_VALIDATION_FAMILY_ERRORS.csv"
    targets_path = args.seal_dir / "E208_VALIDATION_TARGET_MAX_ERRORS.csv"
    thresholds_path = args.seal_dir / "E208_REGISTERED_FAMILY_THRESHOLDS.csv"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    registered_files = {
        risk_path: status.get("risk_table", {}).get("sha256"),
        calibration_path: status.get("calibration_tasks", {}).get("sha256"),
        targets_path: status.get("calibration_targets", {}).get("sha256"),
        thresholds_path: status.get("family_thresholds", {}).get("sha256"),
    }
    if (
        status.get("status") != "PASS_AWAITING_REMOTE_SEAL"
        or int(status.get("n_tasks", -1)) != 224
        or int(status.get("test_perturbed_expression_rows_read", -1)) != 0
        or status.get("risk_formula") != "(4*rank(predicted_magnitude)+rank(mean(zD,zG,zH,zN,zC)))/(5*n_context_tasks)"
        or len(status.get("prediction_inputs", [])) != 5
        or any(not path.is_file() or sha256_file(path) != expected for path, expected in registered_files.items())
    ):
        raise AuditFailure("seal status or registered file hashes failed")

    validation_members: list[np.ndarray] = []
    test_members: list[np.ndarray] = []
    validation_controls = test_controls = None
    validation_tasks = test_tasks = None
    prediction_hash_checks = 0
    for architecture, seed in JOBS:
        vp, vc, vt, vs = read_prediction(args.prediction_root, "validation", architecture, seed)
        tp, tc, tt, ts = read_prediction(args.prediction_root, "test_pretruth", architecture, seed)
        if validation_controls is None:
            validation_controls, validation_tasks = vc, vt
            test_controls, test_tasks = tc, tt
        elif not np.array_equal(vc, validation_controls) or not vt.equals(validation_tasks):
            raise AuditFailure("validation task/control order differs across members")
        elif not np.array_equal(tc, test_controls) or not tt.equals(test_tasks):
            raise AuditFailure("test task/control order differs across members")
        registered = next(
            item for item in status["prediction_inputs"]
            if item["architecture"] == architecture and int(item["seed"]) == seed
        )
        for observed, key in (
            (vs["checkpoint_sha256"], "validation_checkpoint_sha256"),
            (vs["prediction_centroids_sha256"], "validation_prediction_sha256"),
            (ts["checkpoint_sha256"], "test_checkpoint_sha256"),
            (ts["prediction_centroids"]["sha256"], "test_prediction_sha256"),
        ):
            if observed != registered[key]:
                raise AuditFailure(f"registered prediction lineage differs: {architecture}/{seed}/{key}")
            prediction_hash_checks += 1
        validation_members.append(vp)
        test_members.append(tp)
    assert validation_tasks is not None and test_tasks is not None
    assert validation_controls is not None and test_controls is not None

    validation_truth_path = args.validation_cache_dir / "E208_VALIDATION_TRUTH_CENTROIDS.npy"
    validation_status_path = args.validation_cache_dir / "E208_VALIDATION_CACHE_STATUS.json"
    validation_status = json.loads(validation_status_path.read_text(encoding="utf-8"))
    if (
        validation_status.get("status") != "PASS"
        or int(validation_status.get("test_perturbed_expression_rows_read", -1)) != 0
        or sha256_file(validation_truth_path) != validation_status["truth_centroids"]["sha256"]
        or sha256_file(validation_status_path) != status["validation_cache_status_sha256"]
    ):
        raise AuditFailure("validation cache lineage failed")
    competence_path = args.prediction_root / "validation_competence" / "E208_VALIDATION_COMPETENCE_STATUS.json"
    competence = json.loads(competence_path.read_text(encoding="utf-8"))
    if (
        competence.get("status") != "PASS"
        or int(competence.get("test_perturbed_expression_rows_read", -1)) != 0
        or sha256_file(competence_path) != status["competence_status_sha256"]
    ):
        raise AuditFailure("validation competence gate lineage failed")
    validation_truth = np.load(validation_truth_path, allow_pickle=False).astype(np.float64)
    _, validation_lower = family_geometry(validation_members)
    validation_error_sq = sum(
        weight * np.mean((member - validation_truth) ** 2, axis=1)
        for weight, member in zip((0.125, 0.125, 0.125, 0.125, 0.5), validation_members, strict=True)
    )
    validation_error = np.sqrt(validation_error_sq)
    sealed_calibration = pd.read_csv(calibration_path)
    if not validation_tasks.task_id.equals(sealed_calibration.task_id):
        raise AuditFailure("validation task order differs from seal")
    differences = {
        "validation_family_rmse": max_abs(validation_error, sealed_calibration.registered_family_rmse.to_numpy(), "validation family RMSE"),
        "validation_family_lower": max_abs(validation_lower, sealed_calibration.registered_family_lower_bound.to_numpy(), "validation family lower"),
    }
    reconstructed_targets = (
        pd.DataFrame({"condition": validation_tasks.condition, "value": validation_error})
        .groupby("condition", sort=True).value.max().reset_index()
    )
    sealed_targets = pd.read_csv(targets_path)
    if not reconstructed_targets.condition.equals(sealed_targets.condition):
        raise AuditFailure("validation calibration target order differs")
    differences["validation_target_max"] = max_abs(
        reconstructed_targets.value.to_numpy(),
        sealed_targets.max_registered_family_rmse.to_numpy(),
        "validation target maxima",
    )
    conformal_rank = min(len(reconstructed_targets), math.ceil((len(reconstructed_targets) + 1) * 0.9))
    conformal_q = float(np.sort(reconstructed_targets.value.to_numpy())[conformal_rank - 1])
    if conformal_rank != int(status["conformal_rank_1_based"]):
        raise AuditFailure("conformal rank differs")
    differences["conformal_q90"] = abs(conformal_q - float(status["conformal_q90"]))
    if differences["conformal_q90"] > TOLERANCE:
        raise AuditFailure("conformal quantile differs")

    risk = pd.read_csv(risk_path)
    if not test_tasks.task_id.equals(risk.task_id):
        raise AuditFailure("test task order differs from risk table")
    _, family_lower = family_geometry(test_members)
    latent_stack = np.stack(test_members[:4])
    latent_mean = latent_stack.mean(axis=0)
    same_arch = np.sqrt(np.mean((latent_stack - latent_mean[None, :, :]) ** 2, axis=(0, 2)))
    magnitude = np.sqrt(np.mean((latent_mean - test_controls) ** 2, axis=1))
    upper = np.sqrt(conformal_q**2 + family_lower**2)
    differences.update(
        {
            "predicted_magnitude": max_abs(magnitude, risk.predicted_magnitude.to_numpy(), "predicted magnitude"),
            "same_arch_disagreement": max_abs(same_arch, risk.same_arch_seed_disagreement.to_numpy(), "same-architecture disagreement"),
            "family_lower": max_abs(family_lower, risk.registered_family_lower_bound.to_numpy(), "family lower"),
            "family_upper": max_abs(upper, risk.registered_family_conformal_upper.to_numpy(), "family conformal upper"),
        }
    )

    source_status_path = args.source_cache_dir / "E208_SOURCE_EFFECT_CACHE_STATUS.json"
    source_effect_path = args.source_cache_dir / "E208_SOURCE_EFFECTS.npy"
    source_manifest_path = args.source_cache_dir / "E208_SOURCE_EFFECT_MANIFEST.csv"
    source_status = json.loads(source_status_path.read_text(encoding="utf-8"))
    if (
        source_status.get("status") != "PASS"
        or int(source_status.get("test_perturbed_expression_rows_read", -1)) != 0
        or sha256_file(source_effect_path) != source_status["effects"]["sha256"]
        or sha256_file(source_manifest_path) != source_status["manifest"]["sha256"]
        or sha256_file(source_status_path) != status["source_cache_status_sha256"]
    ):
        raise AuditFailure("source-effect cache lineage failed")
    source_effects = np.load(source_effect_path, allow_pickle=False).astype(np.float64)
    source_manifest = pd.read_csv(source_manifest_path)
    raw = {name: [] for name in ("D", "G", "H", "N", "C", "n_source_contexts", "n_source_perturbed_cells")}
    for index, task in enumerate(test_tasks.itertuples(index=False)):
        target_context = f"{task.cell_type}|{task.treatment}"
        sources = source_manifest.loc[
            source_manifest.condition.astype(str).eq(str(task.condition))
            & ~source_manifest.source_context_id.astype(str).eq(target_context)
        ]
        if len(sources) < 2:
            raise AuditFailure(f"fewer than two source contexts for {task.task_id}")
        effects = source_effects[sources.effect_row.to_numpy(np.int64)]
        mean_effect = effects.mean(axis=0)
        raw["D"].append(same_arch[index])
        raw["G"].append(np.sqrt(np.mean((latent_mean[index] - (test_controls[index] + mean_effect)) ** 2)))
        raw["H"].append(np.sqrt(np.mean((effects - mean_effect[None, :]) ** 2)))
        raw["N"].append(-np.log1p(sources.n_perturbed_cells.sum()))
        raw["C"].append(29 - len(sources))
        raw["n_source_contexts"].append(len(sources))
        raw["n_source_perturbed_cells"].append(int(sources.n_perturbed_cells.sum()))
    for feature, values in raw.items():
        differences[feature] = max_abs(np.asarray(values), risk[feature].to_numpy(), feature)

    reconstructed = pd.DataFrame(raw)
    reconstructed["safeconf_original"] = np.nan
    reconstructed["safeconf_m_primary"] = np.nan
    reconstructed["safeconf_oneway_secondary"] = np.nan
    for _, index in test_tasks.groupby(["cell_type", "treatment"], sort=True).groups.items():
        indices = list(index)
        z_values = []
        for feature in ("D", "G", "H", "N", "C"):
            values = reconstructed.loc[indices, feature].to_numpy(float)
            z = (values - values.mean()) / values.std(ddof=0)
            z_values.append(z)
            differences[f"z_{feature}"] = max(
                differences.get(f"z_{feature}", 0.0),
                max_abs(z, risk.loc[indices, f"z_{feature}"].to_numpy(), f"z_{feature}"),
            )
        safeconf = np.mean(np.stack(z_values), axis=0)
        mag_rank = pd.Series(magnitude[indices], index=indices).rank(method="average", ascending=True)
        safe_rank = pd.Series(safeconf, index=indices).rank(method="average", ascending=True)
        n = len(indices)
        combined = (4.0 * mag_rank.to_numpy() + safe_rank.to_numpy()) / (5.0 * n)
        oneway = mag_rank.to_numpy() / n + 0.25 * np.maximum(
            safe_rank.to_numpy() / n - mag_rank.to_numpy() / n, 0.0
        )
        reconstructed.loc[indices, "safeconf_original"] = safeconf
        reconstructed.loc[indices, "safeconf_m_primary"] = combined
        reconstructed.loc[indices, "safeconf_oneway_secondary"] = oneway
    for feature in ("safeconf_original", "safeconf_m_primary", "safeconf_oneway_secondary"):
        differences[feature] = max_abs(reconstructed[feature].to_numpy(), risk[feature].to_numpy(), feature)

    quantiles = np.arange(0.1, 1.0, 0.1)
    thresholds = pd.read_csv(thresholds_path)
    if not np.allclose(quantiles, thresholds["quantile"].to_numpy(), rtol=0.0, atol=1e-15):
        raise AuditFailure("certificate quantiles differ")
    differences["family_thresholds"] = max_abs(
        np.quantile(family_lower, quantiles, method="linear"), thresholds.tau.to_numpy(), "family thresholds"
    )

    audit = {
        "experiment": "E208_jiang24_external_confirmation",
        "stage": "D1_INDEPENDENT_PRETRUTH_RISK_SEAL_AUDIT",
        "status": "PASS",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "n_tasks": len(risk),
        "n_contexts": len(risk[["cell_type", "treatment"]].drop_duplicates()),
        "n_target_genes": int(risk.condition.nunique()),
        "prediction_lineage_hash_checks": prediction_hash_checks,
        "maximum_absolute_differences": differences,
        "maximum_over_all_numeric_checks": float(max(differences.values())),
        "tolerance": TOLERANCE,
        "risk_table_sha256": sha256_file(risk_path),
        "risk_status_sha256": sha256_file(status_path),
        "thresholds_sha256": sha256_file(thresholds_path),
        "test_perturbed_expression_rows_read": 0,
        "target_truth_access": "NOT_AUTHORIZED",
        "next_required_gate": "copy seal and audit into Git, push both remotes, then create separate truth authorization",
    }
    atomic_json(args.output_dir / "E208_PRETRUTH_RISK_SEAL_INDEPENDENT_AUDIT.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
