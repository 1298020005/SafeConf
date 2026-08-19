#!/usr/bin/env python3
"""Run the frozen E201 four-target, four-seed core evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
OUT = ROOT / "docs/实验结果/E201_txpert_multitarget_retraining_20260802"
FREEZE = OUT / "FORMAL_CORE_EVALUATION_FREEZE.md"
RISK_STATUS = OUT / "E201_PRETRUTH_RISK_STATUS.json"
RISK_TABLE = OUT / "tables/E201_PRETRUTH_RISK_FEATURES.csv"
GENERAL_STATUS = OUT / "E201_OFFICIAL_GENERAL_BASELINE_STATUS.json"
RELEASE_STATUS = OUT / "E201_TARGET_TRUTH_RELEASE_STATUS.json"
FINAL = OUT / "formal_core_evaluation"
TABLES = FINAL / "tables"
REPORTS = FINAL / "reports"
FIGURES = FINAL / "figures"
FINAL_STATUS = FINAL / "E201_CORE_FINAL_STATUS.json"

TARGETS = ("K562", "RPE1", "hepg2", "jurkat")
SEEDS = (1, 2, 3, 4)
EXPECTED_TARGET_SAMPLES = {
    "K562": 150_472,
    "RPE1": 67_034,
    "hepg2": 54_911,
    "jurkat": 81_791,
}
EXPECTED_TARGET_TASKS = {
    "K562": (580, 566, 14),
    "RPE1": (467, 416, 51),
    "hepg2": (480, 405, 75),
    "jurkat": (481, 421, 60),
}
EXPECTED_VECTORS = {
    "E201_SEED_CENTROIDS.npy": (4, 2_008, 3_352),
    "E201_FAMILY_CENTROIDS.npy": (2_008, 3_352),
    "E201_CONTROL_CENTROIDS.npy": (2_008, 3_352),
    "E201_SOURCE_TRANSFER_CENTROIDS.npy": (2_008, 3_352),
}
EXPECTED_VAR_ORDER_SHA256 = (
    "d67c176fda6515159421fea6fbaca860240cb6980ccc51745bff619dfec489ca"
)
N_TASKS = 2_008
N_PRIMARY = 1_808
N_SENSITIVITY = 200
N_GENES = 3_352
N_BOOTSTRAP = 5_000
BUDGET = 0.20
MASTER_SEED = 20_260_802
IDENTITY_TOL = 1e-10
LOWER_BOUND_TOL = 1e-12
PRETRUTH_FEATURE_TOL = 5e-6
FAMILY_MEAN_TOL = 2e-6

MAIN_PREDICTORS = (
    "safeconf_e201_risk",
    "predicted_magnitude",
    "family_disagreement",
)
DESCRIPTIVE_PREDICTORS = (
    "safeconf_e201_risk",
    "predicted_magnitude",
    "family_disagreement",
    "family_radius",
    "model_source_gap",
    "source_delta_dispersion",
    "negative_log_source_cells",
    "support_context_deficit",
)
DESCRIPTIVE_OUTCOMES = (
    "family_rms_error",
    "family_centroid_rmse",
    "worst_seed_error",
)


class EvaluationFailure(RuntimeError):
    """Fail-closed E201 formal core-evaluation error."""


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--evaluation-vector-dir", type=Path)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="exercise frozen formulas without opening E201 outcomes",
    )
    args = parser.parse_args()
    if not args.self_test and (
        args.data_root is None or args.evaluation_vector_dir is None
    ):
        parser.error("formal mode requires --data-root and --evaluation-vector-dir")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(label: str) -> int:
    payload = f"E201::{MASTER_SEED}::{label}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


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


def remote_tip(remote: str, branch: str) -> str:
    line = git_text("ls-remote", remote, f"refs/heads/{branch}")
    if not line:
        raise EvaluationFailure(f"missing remote branch: {remote}/{branch}")
    return line.split()[0]


def verify_git_release() -> str:
    required = (
        SCRIPT,
        FREEZE,
        RISK_STATUS,
        RISK_TABLE,
        GENERAL_STATUS,
        RELEASE_STATUS,
    )
    if not all(path.is_file() and tracked_clean(path) for path in required):
        raise EvaluationFailure(
            "formal code/freeze/risk seal/truth release is not tracked and clean"
        )
    branch = git_text("branch", "--show-current")
    head = git_text("rev-parse", "HEAD")
    if not branch:
        raise EvaluationFailure("detached HEAD is not allowed")
    for remote in ("origin", "github"):
        if remote_tip(remote, branch) != head:
            raise EvaluationFailure(f"{remote}/{branch} differs from local HEAD")
    return head


def resolve_data_path(text: str, data_root: Path) -> Path:
    if not text.startswith("DATA/"):
        raise EvaluationFailure(f"path is not DATA-relative: {text}")
    path = (data_root / text[len("DATA/") :]).resolve()
    try:
        path.relative_to(data_root.resolve())
    except ValueError as exc:
        raise EvaluationFailure(f"data path escapes root: {text}") from exc
    return path


def logical_path(path: Path, data_root: Path) -> str:
    resolved = path.resolve()
    for base, prefix in ((ROOT.resolve(), ""), (data_root.resolve(), "DATA/")):
        try:
            return prefix + resolved.relative_to(base).as_posix()
        except ValueError:
            continue
    return path.name


def atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(temporary, path)


def file_record(path: Path, data_root: Path) -> dict[str, Any]:
    return {
        "path": logical_path(path, data_root),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def verify_record(path: Path, record: dict[str, Any], label: str) -> None:
    if (
        not path.is_file()
        or path.stat().st_size != int(record["bytes"])
        or sha256_file(path) != record["sha256"]
    ):
        raise EvaluationFailure(f"{label} file changed: {path}")


def verify_risk_inputs(
    data_root: Path,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict, list[dict[str, Any]]]:
    status_sha = sha256_file(RISK_STATUS)
    status = json.loads(RISK_STATUS.read_text(encoding="utf-8"))
    if (
        status.get("status") != "PASS"
        or int(status.get("n_tasks", -1)) != N_TASKS
        or int(status.get("n_primary_tasks", -1)) != N_PRIMARY
        or int(status.get("target_expression_nonzero_values_seen", -1)) != 0
        or status.get("target_truth_materialized") is not False
        or status.get("target_outcomes_evaluated") is not False
    ):
        raise EvaluationFailure("pretruth risk status failed")
    risk_record = status.get("risk_table", {})
    expected_risk_path = RISK_TABLE.relative_to(ROOT).as_posix()
    if risk_record.get("path") != expected_risk_path:
        raise EvaluationFailure("pretruth risk-table path changed")
    verify_record(RISK_TABLE, risk_record, "pretruth risk table")
    features = pd.read_csv(RISK_TABLE, keep_default_na=True)
    required_columns = {
        "task_id",
        "target",
        "condition",
        "gene",
        "n_target_cells",
        "analysis_stratum",
        "source_mean_delta_row",
        "family_disagreement",
        "family_radius",
        "predicted_magnitude",
        "model_source_gap",
        "source_delta_dispersion",
        "negative_log_source_cells",
        "support_context_deficit",
        "safeconf_e201_risk",
        "z_family_disagreement",
        "z_model_source_gap",
        "z_source_delta_dispersion",
        "z_negative_log_source_cells",
        "z_support_context_deficit",
    }
    if not required_columns.issubset(features.columns):
        missing = sorted(required_columns - set(features.columns))
        raise EvaluationFailure(f"pretruth risk schema missing: {missing}")
    if (
        len(features) != N_TASKS
        or features.task_id.nunique() != N_TASKS
        or not np.array_equal(
            features.source_mean_delta_row.to_numpy(int), np.arange(N_TASKS)
        )
        or int(features.analysis_stratum.eq("primary_ge30").sum()) != N_PRIMARY
        or int(features.analysis_stratum.eq("sensitivity_10_29").sum()) != N_SENSITIVITY
        or tuple(pd.unique(features.target.astype(str))) != TARGETS
    ):
        raise EvaluationFailure("pretruth risk task ordering changed")
    for target, (total, primary, sensitivity) in EXPECTED_TARGET_TASKS.items():
        block = features.loc[features.target.eq(target)]
        observed = (
            len(block),
            int(block.analysis_stratum.eq("primary_ge30").sum()),
            int(block.analysis_stratum.eq("sensitivity_10_29").sum()),
        )
        if observed != (total, primary, sensitivity):
            raise EvaluationFailure(f"task inventory changed: {target}/{observed}")
    finite_columns = list(DESCRIPTIVE_PREDICTORS) + [
        "z_family_disagreement",
        "z_model_source_gap",
        "z_source_delta_dispersion",
        "z_negative_log_source_cells",
        "z_support_context_deficit",
    ]
    if not np.isfinite(features[finite_columns].to_numpy(float)).all():
        raise EvaluationFailure("non-finite pretruth risk feature")
    recomputed_risk = features[
        [
            "z_family_disagreement",
            "z_model_source_gap",
            "z_source_delta_dispersion",
            "z_negative_log_source_cells",
            "z_support_context_deficit",
        ]
    ].mean(axis=1)
    risk_identity = float(np.max(np.abs(recomputed_risk - features.safeconf_e201_risk)))
    if risk_identity > 1e-12:
        raise EvaluationFailure("pretruth SafeConf risk identity changed")

    vector_records = status.get("vector_files", [])
    by_name = {Path(record["path"]).name: record for record in vector_records}
    if set(by_name) != set(EXPECTED_VECTORS):
        raise EvaluationFailure("pretruth centroid vector family changed")
    vectors: dict[str, np.ndarray] = {}
    inputs = [
        {
            **file_record(RISK_STATUS, data_root),
            "role": "pretruth_risk_status",
        },
        {**file_record(RISK_TABLE, data_root), "role": "pretruth_risk_table"},
    ]
    for filename, shape in EXPECTED_VECTORS.items():
        record = by_name[filename]
        path = resolve_data_path(record["path"], data_root)
        verify_record(path, record, filename)
        values = np.load(path, mmap_mode="r")
        if values.shape != shape or str(values.dtype) != "float32":
            raise EvaluationFailure(f"pretruth vector contract changed: {filename}")
        if not np.isfinite(values).all():
            raise EvaluationFailure(f"non-finite pretruth vector: {filename}")
        vectors[filename] = values
        inputs.append({**file_record(path, data_root), "role": filename})
    family_residual = float(
        np.max(
            np.abs(
                np.asarray(vectors["E201_SEED_CENTROIDS.npy"]).mean(axis=0)
                - np.asarray(vectors["E201_FAMILY_CENTROIDS.npy"])
            )
        )
    )
    if family_residual > FAMILY_MEAN_TOL:
        raise EvaluationFailure("sealed family centroid is not the four-seed mean")
    status["verified_status_sha256"] = status_sha
    status["verified_risk_identity_max_abs_residual"] = risk_identity
    status["verified_family_mean_max_abs_residual"] = family_residual
    return features, vectors, status, inputs


def verify_general_baseline(
    data_root: Path, risk_status: dict
) -> tuple[np.ndarray, dict, list[dict[str, Any]]]:
    status_sha = sha256_file(GENERAL_STATUS)
    status = json.loads(GENERAL_STATUS.read_text(encoding="utf-8"))
    equivalence = status.get("e200_official_code_equivalence", {})
    if (
        status.get("status") != "PASS"
        or int(status.get("n_tasks", -1)) != N_TASKS
        or int(status.get("n_primary_tasks", -1)) != N_PRIMARY
        or int(status.get("target_perturbed_expression_rows_opened", -1)) != 0
        or status.get("target_truth_materialized") is not False
        or status.get("target_outcomes_evaluated") is not False
        or status.get("pretruth_risk_status_sha256") != sha256_file(RISK_STATUS)
        or status.get("pretruth_risk_table_sha256")
        != risk_status["risk_table"]["sha256"]
        or equivalence.get("passed") is not True
        or int(equivalence.get("tasks_exceeding_tolerance", -1)) != 0
        or float(equivalence.get("maximum_absolute_delta_residual", float("inf")))
        > 5e-6
        or equivalence.get("target_truth_opened") is not False
    ):
        raise EvaluationFailure("pretruth official general-baseline seal failed")
    records = status.get("vector_files", [])
    by_name = {Path(record["path"]).name: record for record in records}
    expected_names = {
        "E201_OFFICIAL_GENERAL_BASELINE_WEIGHTED_DELTAS.npy",
        "E201_OFFICIAL_GENERAL_BASELINE_CENTROIDS.npy",
    }
    if set(by_name) != expected_names:
        raise EvaluationFailure("official general-baseline vector family changed")
    inputs = [
        {
            **file_record(GENERAL_STATUS, data_root),
            "role": "official_general_baseline_status",
        }
    ]
    arrays = {}
    for filename in sorted(expected_names):
        record = by_name[filename]
        path = resolve_data_path(record["path"], data_root)
        verify_record(path, record, filename)
        values = np.load(path, mmap_mode="r")
        if values.shape != (N_TASKS, N_GENES) or str(values.dtype) != "float32":
            raise EvaluationFailure(f"official baseline vector changed: {filename}")
        if not np.isfinite(values).all():
            raise EvaluationFailure(f"non-finite official baseline: {filename}")
        arrays[filename] = values
        inputs.append({**file_record(path, data_root), "role": filename})
    for record in status.get("tracked_outputs", []):
        path = ROOT / record["path"]
        if not tracked_clean(path):
            raise EvaluationFailure(
                f"official baseline tracked output is not clean: {path}"
            )
        verify_record(path, record, "official baseline tracked output")
        inputs.append(
            {**file_record(path, data_root), "role": "official_baseline_audit"}
        )
    status["verified_status_sha256"] = status_sha
    return (
        arrays["E201_OFFICIAL_GENERAL_BASELINE_CENTROIDS.npy"],
        status,
        inputs,
    )


def verify_truth_release(
    data_root: Path, risk_status: dict, general_status: dict
) -> tuple[dict[str, dict[str, Any]], dict, list[dict[str, Any]]]:
    release = json.loads(RELEASE_STATUS.read_text(encoding="utf-8"))
    risk_status_sha = sha256_file(RISK_STATUS)
    general_status_sha = sha256_file(GENERAL_STATUS)
    general_centroid_records = [
        record
        for record in general_status["vector_files"]
        if Path(record["path"]).name == "E201_OFFICIAL_GENERAL_BASELINE_CENTROIDS.npy"
    ]
    if len(general_centroid_records) != 1:
        raise EvaluationFailure("official general-baseline centroid record missing")
    general_centroid_sha = general_centroid_records[0]["sha256"]
    if (
        release.get("status") != "PASS"
        or int(release.get("n_targets", -1)) != 4
        or release.get("pretruth_risk_sealed_before_release") is not True
        or release.get("pretruth_risk_status_sha256") != risk_status_sha
        or release.get("pretruth_official_general_baseline_sealed_before_release")
        is not True
        or release.get("pretruth_official_general_baseline_status_sha256")
        != general_status_sha
        or release.get("official_general_baseline_centroid_sha256")
        != general_centroid_sha
    ):
        raise EvaluationFailure("target-truth release status failed")
    records = release.get("records", [])
    by_target = {record.get("target"): record for record in records}
    if set(by_target) != set(TARGETS):
        raise EvaluationFailure("target-truth release family changed")
    inputs = [
        {**file_record(RELEASE_STATUS, data_root), "role": "truth_release_status"}
    ]
    verified: dict[str, dict[str, Any]] = {}
    for target in TARGETS:
        record = by_target[target]
        truth_path = resolve_data_path(record["truth_path"], data_root)
        manifest_path = resolve_data_path(record["truth_manifest_path"], data_root)
        if (
            int(record.get("n_samples", -1)) != EXPECTED_TARGET_SAMPLES[target]
            or truth_path.stat().st_size != int(record["truth_bytes"])
            or sha256_file(truth_path) != record["truth_sha256"]
            or sha256_file(manifest_path) != record["truth_manifest_sha256"]
        ):
            raise EvaluationFailure(f"released target record changed: {target}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") != "SEALED_TARGET_TRUTH"
            or manifest.get("target") != target
            or int(manifest.get("n_samples", -1)) != EXPECTED_TARGET_SAMPLES[target]
            or int(manifest.get("n_genes", -1)) != N_GENES
            or manifest.get("var_order_sha256") != EXPECTED_VAR_ORDER_SHA256
            or manifest.get("risk_status_sha256") != risk_status_sha
            or manifest.get("risk_table_sha256") != risk_status["risk_table"]["sha256"]
            or manifest.get("official_general_baseline_status_sha256")
            != general_status_sha
            or manifest.get("official_general_baseline_centroid_sha256")
            != general_centroid_sha
            or manifest.get("alignment")
            != {
                "condition": True,
                "cell_type": True,
                "experimental_batch": True,
                "row_index": True,
            }
            or int(manifest.get("target_expression_rows_opened", -1))
            != EXPECTED_TARGET_SAMPLES[target]
        ):
            raise EvaluationFailure(f"released target manifest failed: {target}")
        verify_record(truth_path, manifest["truth_file"], f"target truth {target}")
        observations_path = truth_path.parent / "observations.csv"
        shared_manifest_path = truth_path.parent / "E201_SHARED_TARGET_MANIFEST.json"
        if (
            not observations_path.is_file()
            or not shared_manifest_path.is_file()
            or sha256_file(shared_manifest_path)
            != manifest["shared_pretruth_manifest_sha256"]
        ):
            raise EvaluationFailure(f"shared target inputs changed: {target}")
        truth = np.load(truth_path, mmap_mode="r")
        if truth.shape != (EXPECTED_TARGET_SAMPLES[target], N_GENES):
            raise EvaluationFailure(f"released truth shape changed: {target}")
        if str(truth.dtype) != "float32" or not np.isfinite(truth).all():
            raise EvaluationFailure(f"released truth values failed: {target}")
        observations = pd.read_csv(observations_path, keep_default_na=False)
        if (
            len(observations) != EXPECTED_TARGET_SAMPLES[target]
            or observations.row_index.tolist() != list(range(len(observations)))
            or set(observations.cell_type.astype(str)) != {target}
        ):
            raise EvaluationFailure(f"released observation alignment failed: {target}")
        verified[target] = {
            "truth": truth,
            "observations": observations,
            "record": record,
            "manifest": manifest,
        }
        inputs.extend(
            [
                {**file_record(truth_path, data_root), "role": f"truth_{target}"},
                {
                    **file_record(manifest_path, data_root),
                    "role": f"truth_manifest_{target}",
                },
                {
                    **file_record(observations_path, data_root),
                    "role": f"observations_{target}",
                },
                {
                    **file_record(shared_manifest_path, data_root),
                    "role": f"shared_manifest_{target}",
                },
            ]
        )
    return verified, release, inputs


def condition_from_label(target: str, label: str) -> str:
    prefix = f"{target}_"
    suffix = "_1+1"
    if not label.startswith(prefix) or not label.endswith(suffix):
        raise EvaluationFailure(f"unexpected target label: {label}")
    condition = label[len(prefix) : -len(suffix)]
    if not condition.endswith("+ctrl"):
        raise EvaluationFailure(f"not a single-perturbation label: {label}")
    return condition


def rmse(left: np.ndarray, right: np.ndarray) -> float:
    difference = np.asarray(left, dtype=np.float64) - np.asarray(
        right, dtype=np.float64
    )
    return float(np.sqrt(np.mean(np.square(difference))))


def compute_task_metrics(
    features: pd.DataFrame,
    vectors: dict[str, np.ndarray],
    general_centroids: np.ndarray,
    truth_release: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    seeds = vectors["E201_SEED_CENTROIDS.npy"]
    families = vectors["E201_FAMILY_CENTROIDS.npy"]
    controls = vectors["E201_CONTROL_CENTROIDS.npy"]
    sources = vectors["E201_SOURCE_TRANSFER_CENTROIDS.npy"]
    truth_centroids = np.empty((N_TASKS, N_GENES), dtype=np.float32)
    task_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []

    for target in TARGETS:
        observations = truth_release[target]["observations"]
        truth = truth_release[target]["truth"]
        conditions = observations.pert_cond_name.astype(str).map(
            lambda label: condition_from_label(target, label)
        )
        groups = conditions.groupby(conditions, sort=False).groups
        target_indices = np.flatnonzero(features.target.to_numpy() == target)
        for task_index in target_indices:
            feature = features.iloc[task_index]
            condition = str(feature.condition)
            if condition not in groups:
                raise EvaluationFailure(f"missing target truth task: {feature.task_id}")
            cell_indices = np.asarray(groups[condition], dtype=int)
            if len(cell_indices) != int(feature.n_target_cells):
                raise EvaluationFailure(
                    f"target truth cell count changed: {feature.task_id}"
                )
            truth_center = np.asarray(truth[cell_indices], dtype=np.float64).mean(
                axis=0
            )
            if not np.isfinite(truth_center).all():
                raise EvaluationFailure(f"non-finite truth centroid: {feature.task_id}")
            truth_centroids[task_index] = truth_center.astype(np.float32)

            seed_vectors = np.asarray(seeds[:, task_index, :], dtype=np.float64)
            family = np.asarray(families[task_index], dtype=np.float64)
            control = np.asarray(controls[task_index], dtype=np.float64)
            source = np.asarray(sources[task_index], dtype=np.float64)
            general = np.asarray(general_centroids[task_index], dtype=np.float64)
            seed_errors = np.sqrt(
                np.mean(np.square(seed_vectors - truth_center[None, :]), axis=1)
            )
            centroid_error = rmse(family, truth_center)
            family_rms = float(np.sqrt(np.mean(np.square(seed_errors))))
            disagreement = float(
                np.sqrt(np.mean(np.square(seed_vectors - family[None, :])))
            )
            predicted_magnitude = rmse(family, control)
            model_source_gap = rmse(family, source)
            identity_residual = float(
                family_rms**2 - centroid_error**2 - disagreement**2
            )
            lower_violation = bool(family_rms < disagreement - LOWER_BOUND_TOL)
            task_rows.append(
                {
                    "task_id": feature.task_id,
                    "target": target,
                    "condition": condition,
                    "gene": feature.gene,
                    "analysis_stratum": feature.analysis_stratum,
                    "n_target_cells": int(feature.n_target_cells),
                    "family_centroid_rmse": centroid_error,
                    "family_rms_error": family_rms,
                    "worst_seed_error": float(seed_errors.max()),
                    "control_error": rmse(control, truth_center),
                    "official_general_baseline_error": rmse(general, truth_center),
                    "source_transfer_error": rmse(source, truth_center),
                    "recomputed_family_disagreement": disagreement,
                    "family_rms_sq_identity_residual": identity_residual,
                    "family_rms_lower_bound_violation": lower_violation,
                    "pretruth_disagreement_abs_residual": abs(
                        disagreement - float(feature.family_disagreement)
                    ),
                    "pretruth_magnitude_abs_residual": abs(
                        predicted_magnitude - float(feature.predicted_magnitude)
                    ),
                    "pretruth_model_source_gap_abs_residual": abs(
                        model_source_gap - float(feature.model_source_gap)
                    ),
                }
            )
            for seed, error in zip(SEEDS, seed_errors):
                seed_rows.append(
                    {
                        "task_id": feature.task_id,
                        "target": target,
                        "condition": condition,
                        "analysis_stratum": feature.analysis_stratum,
                        "seed": seed,
                        "seed_centroid_rmse": float(error),
                    }
                )

    task_metrics = pd.DataFrame(task_rows)
    seed_metrics = pd.DataFrame(seed_rows)
    if (
        len(task_metrics) != N_TASKS
        or task_metrics.task_id.tolist() != features.task_id.tolist()
        or len(seed_metrics) != len(SEEDS) * N_TASKS
        or not np.isfinite(truth_centroids).all()
        or not np.isfinite(
            task_metrics.select_dtypes(include=[np.number]).to_numpy(float)
        ).all()
    ):
        raise EvaluationFailure("formal task metric contract failed")
    return task_metrics, seed_metrics, truth_centroids


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    keep = np.isfinite(left) & np.isfinite(right)
    if (
        keep.sum() < 4
        or np.unique(left[keep]).size < 2
        or np.unique(right[keep]).size < 2
    ):
        return float("nan")
    return float(
        np.corrcoef(
            rankdata(left[keep], method="average"),
            rankdata(right[keep], method="average"),
        )[0, 1]
    )


def partial_spearman(
    predictor: np.ndarray, outcome: np.ndarray, covariate: np.ndarray
) -> float:
    predictor = np.asarray(predictor, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    covariate = np.asarray(covariate, dtype=float)
    keep = np.isfinite(predictor) & np.isfinite(outcome) & np.isfinite(covariate)
    if keep.sum() < 5:
        return float("nan")
    ranked_x = rankdata(predictor[keep], method="average")
    ranked_y = rankdata(outcome[keep], method="average")
    ranked_z = rankdata(covariate[keep], method="average")
    if np.unique(ranked_z).size < 2:
        return float("nan")
    design = np.column_stack([np.ones(len(ranked_z)), ranked_z])
    residual_x = ranked_x - design @ np.linalg.lstsq(design, ranked_x, rcond=None)[0]
    residual_y = ranked_y - design @ np.linalg.lstsq(design, ranked_y, rcond=None)[0]
    if np.std(residual_x) <= 0 or np.std(residual_y) <= 0:
        return float("nan")
    return float(np.corrcoef(residual_x, residual_y)[0, 1])


def tie_values(task_ids: pd.Series | np.ndarray, occurrences: np.ndarray) -> np.ndarray:
    values = []
    for task_id, occurrence in zip(map(str, task_ids), occurrences):
        payload = f"E201\0{task_id}\0{int(occurrence)}".encode("utf-8")
        values.append(int(hashlib.sha256(payload).hexdigest()[:16], 16))
    return np.asarray(values, dtype=np.uint64)


def utility_arrays(
    risk: np.ndarray,
    outcome: np.ndarray,
    task_ids: pd.Series | np.ndarray,
    occurrences: np.ndarray | None = None,
) -> dict[str, float]:
    risk = np.asarray(risk, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    if occurrences is None:
        occurrences = np.zeros(len(risk), dtype=np.int64)
    keep = np.isfinite(risk) & np.isfinite(outcome)
    risk = risk[keep]
    outcome = outcome[keep]
    kept_tasks = np.asarray(list(map(str, task_ids)))[keep]
    occurrences = np.asarray(occurrences, dtype=np.int64)[keep]
    if len(risk) < 5:
        raise EvaluationFailure("too few finite tasks for review utility")
    ties = tie_values(kept_tasks, occurrences)
    n_select = int(math.ceil(BUDGET * len(risk)))
    risk_order = np.lexsort((ties, -risk))[:n_select]
    oracle_order = np.lexsort((ties, -outcome))[:n_select]
    selected = float(outcome[risk_order].mean())
    oracle = float(outcome[oracle_order].mean())
    overall = float(outcome.mean())
    denominator = oracle - overall
    utility = (
        float((selected - overall) / denominator)
        if denominator > 1e-15
        else float("nan")
    )
    return {
        "budget": BUDGET,
        "n_tasks": len(risk),
        "n_selected": n_select,
        "high_error_capture": float(
            len(set(risk_order.tolist()) & set(oracle_order.tolist())) / n_select
        ),
        "random_expected_capture": float(n_select / len(risk)),
        "selected_mean_error": selected,
        "overall_mean_error": overall,
        "error_lift": float(selected / overall),
        "oracle_mean_error": oracle,
        "oracle_normalized_utility": utility,
    }


def cluster_resample_indices(
    frame: pd.DataFrame, rng: np.random.Generator
) -> np.ndarray:
    clusters = sorted(frame.condition.astype(str).unique())
    members = [
        np.flatnonzero(frame.condition.astype(str).to_numpy() == cluster)
        for cluster in clusters
    ]
    chosen = rng.integers(0, len(clusters), len(clusters))
    return np.concatenate([members[index] for index in chosen])


def scope_statistics(frame: pd.DataFrame) -> dict[str, float]:
    outcome = frame.family_rms_error.to_numpy(float)
    safeconf = frame.safeconf_e201_risk.to_numpy(float)
    magnitude = frame.predicted_magnitude.to_numpy(float)
    disagreement = frame.family_disagreement.to_numpy(float)
    occurrences = frame.groupby("task_id", sort=False).cumcount().to_numpy(int)
    safe_utility = utility_arrays(safeconf, outcome, frame.task_id, occurrences)[
        "oracle_normalized_utility"
    ]
    magnitude_utility = utility_arrays(magnitude, outcome, frame.task_id, occurrences)[
        "oracle_normalized_utility"
    ]
    return {
        "rho_safeconf": spearman(safeconf, outcome),
        "rho_magnitude": spearman(magnitude, outcome),
        "rho_disagreement": spearman(disagreement, outcome),
        "partial_safeconf_given_magnitude": partial_spearman(
            safeconf, outcome, magnitude
        ),
        "utility_safeconf": safe_utility,
        "utility_magnitude": magnitude_utility,
        "delta_utility_safeconf_minus_magnitude": (safe_utility - magnitude_utility),
        "delta_rho_safeconf_minus_magnitude": (
            spearman(safeconf, outcome) - spearman(magnitude, outcome)
        ),
        "mean_delta_family_centroid_minus_control": float(
            (frame.family_centroid_rmse - frame.control_error).mean()
        ),
        "mean_delta_family_centroid_minus_official_general": float(
            (frame.family_centroid_rmse - frame.official_general_baseline_error).mean()
        ),
        "mean_delta_family_centroid_minus_source": float(
            (frame.family_centroid_rmse - frame.source_transfer_error).mean()
        ),
    }


def summarize_draws(
    values: np.ndarray, point: float, label: str
) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < math.ceil(0.95 * N_BOOTSTRAP):
        raise EvaluationFailure(f"too few valid bootstrap draws: {label}")
    return {
        "estimate": float(point),
        "ci95_lower": float(np.quantile(values, 0.025)),
        "ci95_upper": float(np.quantile(values, 0.975)),
        "bootstrap_valid": len(values),
    }


def bootstrap_scope(
    frame: pd.DataFrame,
    scope: str,
    n_bootstrap: int = N_BOOTSTRAP,
) -> tuple[dict[str, float], pd.DataFrame]:
    point = scope_statistics(frame)
    rng = np.random.default_rng(stable_seed(f"cluster-bootstrap::{scope}"))
    rows = []
    for replicate in range(n_bootstrap):
        indices = cluster_resample_indices(frame, rng)
        sampled = frame.iloc[indices].copy()
        values = scope_statistics(sampled)
        rows.append({"replicate": replicate, **values})
    draws = pd.DataFrame(rows)
    return point, draws


def build_inference_tables(
    primary: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    association_rows = []
    partial_rows = []
    utility_rows = []
    incremental_rows = []
    baseline_rows = []
    pooled_draws = None
    predictor_fields = {
        "safeconf_e201_risk": "rho_safeconf",
        "predicted_magnitude": "rho_magnitude",
        "family_disagreement": "rho_disagreement",
    }
    utility_fields = {
        "safeconf_e201_risk": "utility_safeconf",
        "predicted_magnitude": "utility_magnitude",
    }
    scopes = [("pooled", primary)] + [
        (target, primary.loc[primary.target.eq(target)].copy()) for target in TARGETS
    ]
    for scope, frame in scopes:
        point, draws = bootstrap_scope(frame, scope)
        if scope == "pooled":
            pooled_draws = draws.copy()
            pooled_draws.insert(0, "scope", scope)
        for predictor, field in predictor_fields.items():
            association_rows.append(
                {
                    "scope": scope,
                    "predictor": predictor,
                    "outcome": "family_rms_error",
                    "n_tasks": len(frame),
                    "n_condition_clusters": frame.condition.nunique(),
                    "bootstrap_unit": "perturbation_condition",
                    **summarize_draws(
                        draws[field].to_numpy(), point[field], f"{scope}/{field}"
                    ),
                }
            )
        field = "partial_safeconf_given_magnitude"
        partial_rows.append(
            {
                "scope": scope,
                "predictor": "safeconf_e201_risk",
                "outcome": "family_rms_error",
                "covariate": "predicted_magnitude",
                "n_tasks": len(frame),
                "n_condition_clusters": frame.condition.nunique(),
                "bootstrap_unit": "perturbation_condition",
                **summarize_draws(
                    draws[field].to_numpy(), point[field], f"{scope}/{field}"
                ),
            }
        )
        for predictor, field in utility_fields.items():
            utility_point = utility_arrays(
                frame[predictor].to_numpy(float),
                frame.family_rms_error.to_numpy(float),
                frame.task_id,
            )
            utility_interval = summarize_draws(
                draws[field].to_numpy(), point[field], f"{scope}/{field}"
            )
            utility_rows.append(
                {
                    "scope": scope,
                    "predictor": predictor,
                    "outcome": "family_rms_error",
                    "n_condition_clusters": frame.condition.nunique(),
                    "bootstrap_unit": "perturbation_condition",
                    **utility_point,
                    "utility_ci95_lower": utility_interval["ci95_lower"],
                    "utility_ci95_upper": utility_interval["ci95_upper"],
                    "bootstrap_valid": utility_interval["bootstrap_valid"],
                }
            )
        for measure, field in (
            (
                "delta_oracle_normalized_utility",
                "delta_utility_safeconf_minus_magnitude",
            ),
            ("delta_spearman", "delta_rho_safeconf_minus_magnitude"),
        ):
            incremental_rows.append(
                {
                    "scope": scope,
                    "estimator": "safeconf_e201_risk",
                    "baseline": "predicted_magnitude",
                    "outcome": "family_rms_error",
                    "measure": measure,
                    "n_tasks": len(frame),
                    "n_condition_clusters": frame.condition.nunique(),
                    "bootstrap_unit": "perturbation_condition",
                    **summarize_draws(
                        draws[field].to_numpy(), point[field], f"{scope}/{field}"
                    ),
                }
            )
        for baseline, field, column in (
            (
                "official_general_baseline",
                "mean_delta_family_centroid_minus_official_general",
                "official_general_baseline_error",
            ),
            (
                "batch_matched_control",
                "mean_delta_family_centroid_minus_control",
                "control_error",
            ),
            (
                "source_transfer",
                "mean_delta_family_centroid_minus_source",
                "source_transfer_error",
            ),
        ):
            delta = frame.family_centroid_rmse - frame[column]
            baseline_rows.append(
                {
                    "scope": scope,
                    "predictor": "four_seed_family_centroid",
                    "baseline": baseline,
                    "outcome": "centroid_rmse",
                    "n_tasks": len(frame),
                    "n_condition_clusters": frame.condition.nunique(),
                    "predictor_mean_error": float(frame.family_centroid_rmse.mean()),
                    "baseline_mean_error": float(frame[column].mean()),
                    "task_win_rate": float((delta < 0).mean()),
                    "bootstrap_unit": "perturbation_condition",
                    **summarize_draws(
                        draws[field].to_numpy(), point[field], f"{scope}/{field}"
                    ),
                }
            )
    if pooled_draws is None:
        raise EvaluationFailure("pooled bootstrap draws were not generated")
    return (
        pd.DataFrame(association_rows),
        pd.DataFrame(partial_rows),
        pd.DataFrame(utility_rows),
        pd.DataFrame(incremental_rows),
        pd.DataFrame(baseline_rows),
        pooled_draws,
    )


def descriptive_associations(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes = [("pooled", frame)] + [
        (target, frame.loc[frame.target.eq(target)]) for target in TARGETS
    ]
    for scope, block in scopes:
        for predictor in DESCRIPTIVE_PREDICTORS:
            for outcome in DESCRIPTIVE_OUTCOMES:
                rows.append(
                    {
                        "scope": scope,
                        "stratum": (
                            str(block.analysis_stratum.iloc[0])
                            if block.analysis_stratum.nunique() == 1
                            else "mixed"
                        ),
                        "predictor": predictor,
                        "outcome": outcome,
                        "n_tasks": len(block),
                        "spearman": spearman(
                            block[predictor].to_numpy(float),
                            block[outcome].to_numpy(float),
                        ),
                        "inference": "descriptive_point_only",
                    }
                )
    return pd.DataFrame(rows)


def target_error_summary(
    task_metrics: pd.DataFrame, seed_metrics: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for stratum in ("primary_ge30", "sensitivity_10_29"):
        for target in TARGETS:
            block = task_metrics.loc[
                task_metrics.analysis_stratum.eq(stratum)
                & task_metrics.target.eq(target)
            ]
            rows.append(
                {
                    "stratum": stratum,
                    "target": target,
                    "predictor": "four_seed_family",
                    "n_tasks": len(block),
                    "family_centroid_rmse_mean": float(
                        block.family_centroid_rmse.mean()
                    ),
                    "family_rms_error_mean": float(block.family_rms_error.mean()),
                    "worst_seed_error_mean": float(block.worst_seed_error.mean()),
                    "seed_centroid_rmse_mean": np.nan,
                    "control_error_mean": float(block.control_error.mean()),
                    "official_general_baseline_error_mean": float(
                        block.official_general_baseline_error.mean()
                    ),
                    "source_transfer_error_mean": float(
                        block.source_transfer_error.mean()
                    ),
                }
            )
            for seed in SEEDS:
                seed_block = seed_metrics.loc[
                    seed_metrics.analysis_stratum.eq(stratum)
                    & seed_metrics.target.eq(target)
                    & seed_metrics.seed.eq(seed)
                ]
                rows.append(
                    {
                        "stratum": stratum,
                        "target": target,
                        "predictor": f"seed_{seed}",
                        "n_tasks": len(seed_block),
                        "family_centroid_rmse_mean": np.nan,
                        "family_rms_error_mean": np.nan,
                        "worst_seed_error_mean": np.nan,
                        "seed_centroid_rmse_mean": float(
                            seed_block.seed_centroid_rmse.mean()
                        ),
                        "control_error_mean": np.nan,
                        "official_general_baseline_error_mean": np.nan,
                        "source_transfer_error_mean": np.nan,
                    }
                )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append("NA" if not np.isfinite(value) else f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def make_figure(
    primary: pd.DataFrame,
    associations: pd.DataFrame,
    utilities: pd.DataFrame,
    path: Path,
) -> None:
    colors = {
        "K562": "#3B6FB6",
        "RPE1": "#D07A2D",
        "hepg2": "#5A8F63",
        "jurkat": "#8A6BB1",
    }
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Noto Sans CJK SC", "DejaVu Sans"],
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "text.color": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.4), facecolor="white")
    for target in TARGETS:
        block = primary.loc[primary.target.eq(target)]
        axes[0, 0].scatter(
            block.safeconf_e201_risk,
            block.family_rms_error,
            s=11,
            alpha=0.48,
            linewidths=0,
            color=colors[target],
            label=target,
        )
    axes[0, 0].set_xlabel("Frozen SafeConf risk")
    axes[0, 0].set_ylabel("Four-seed family RMS error")
    axes[0, 0].set_title("A  Pretruth risk and family error", loc="left")
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)

    deciles = primary.copy()
    deciles["risk_decile"] = (
        pd.qcut(
            deciles.safeconf_e201_risk.rank(method="first"),
            10,
            labels=False,
        )
        + 1
    )
    decile_summary = deciles.groupby("risk_decile").family_rms_error.agg(
        mean="mean", sem="sem"
    )
    axes[0, 1].errorbar(
        decile_summary.index,
        decile_summary["mean"],
        yerr=decile_summary["sem"],
        marker="o",
        color="#3B6FB6",
        ecolor="#777777",
        capsize=2,
        linewidth=1.35,
    )
    axes[0, 1].set_xlabel("SafeConf risk decile")
    axes[0, 1].set_ylabel("Mean family RMS error")
    axes[0, 1].set_title("B  Error across fixed risk deciles", loc="left")

    pooled_utility = (
        utilities.loc[utilities.scope.eq("pooled")]
        .set_index("predictor")
        .loc[["safeconf_e201_risk", "predicted_magnitude"]]
    )
    values = pooled_utility.oracle_normalized_utility.to_numpy(float)
    lower = values - pooled_utility.utility_ci95_lower.to_numpy(float)
    upper = pooled_utility.utility_ci95_upper.to_numpy(float) - values
    axes[1, 0].bar([0, 1], values, color=["#3B6FB6", "#B6B6B6"], width=0.6)
    axes[1, 0].errorbar(
        [0, 1],
        values,
        yerr=np.vstack([np.maximum(lower, 0), np.maximum(upper, 0)]),
        fmt="none",
        ecolor="#333333",
        capsize=3,
        linewidth=0.9,
    )
    axes[1, 0].axhline(0, color="#777777", linewidth=0.8)
    axes[1, 0].set_xticks([0, 1], ["SafeConf", "Magnitude"])
    axes[1, 0].set_ylabel("Oracle-normalized utility")
    axes[1, 0].set_title("C  Fixed 20% review budget", loc="left")

    forest = (
        associations.loc[
            associations.scope.isin(TARGETS)
            & associations.predictor.eq("safeconf_e201_risk")
        ]
        .set_index("scope")
        .loc[list(TARGETS)]
    )
    y = np.arange(len(TARGETS))
    estimate = forest.estimate.to_numpy(float)
    axes[1, 1].errorbar(
        estimate,
        y,
        xerr=np.vstack(
            [
                estimate - forest.ci95_lower.to_numpy(float),
                forest.ci95_upper.to_numpy(float) - estimate,
            ]
        ),
        fmt="o",
        color="#3B6FB6",
        ecolor="#555555",
        capsize=2,
        markersize=4.5,
    )
    axes[1, 1].axvline(0, color="#777777", linestyle="--", linewidth=0.8)
    axes[1, 1].set_yticks(y, TARGETS)
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlabel("Spearman with family RMS error")
    axes[1, 1].set_title("D  Target-specific estimates", loc="left")

    for axis in axes.ravel():
        axis.set_facecolor("white")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E8E8E8", linewidth=0.55, zorder=0)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_gates(
    task_frame: pd.DataFrame,
    risk_status: dict,
    general_status: dict,
    associations: pd.DataFrame,
    partials: pd.DataFrame,
    utilities: pd.DataFrame,
    increments: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    identity_max = float(task_frame.family_rms_sq_identity_residual.abs().max())
    lower_violations = int(task_frame.family_rms_lower_bound_violation.sum())
    pretruth_max = float(
        task_frame[
            [
                "pretruth_disagreement_abs_residual",
                "pretruth_magnitude_abs_residual",
                "pretruth_model_source_gap_abs_residual",
            ]
        ]
        .to_numpy(float)
        .max()
    )
    pooled_assoc = associations.loc[
        associations.scope.eq("pooled")
        & associations.predictor.eq("safeconf_e201_risk")
    ].iloc[0]
    pooled_partial = partials.loc[partials.scope.eq("pooled")].iloc[0]
    pooled_utility = utilities.loc[
        utilities.scope.eq("pooled") & utilities.predictor.eq("safeconf_e201_risk")
    ].iloc[0]
    pooled_increment = increments.loc[
        increments.scope.eq("pooled")
        & increments.measure.eq("delta_oracle_normalized_utility")
    ].iloc[0]
    input_integrity = bool(
        len(task_frame) == N_TASKS
        and int(task_frame.analysis_stratum.eq("primary_ge30").sum()) == N_PRIMARY
        and risk_status["verified_family_mean_max_abs_residual"] <= FAMILY_MEAN_TOL
        and pretruth_max < PRETRUTH_FEATURE_TOL
        and general_status["e200_official_code_equivalence"]["passed"] is True
    )
    certificate = bool(identity_max <= IDENTITY_TOL and lower_violations == 0)
    routing = bool(
        pooled_assoc.ci95_lower > 0 and pooled_utility.utility_ci95_lower > 0
    )
    incremental = bool(pooled_partial.ci95_lower > 0 or pooled_increment.ci95_lower > 0)
    target_reporting = bool(
        set(associations.scope) >= set(TARGETS)
        and associations.loc[
            associations.scope.isin(TARGETS)
            & associations.predictor.eq("safeconf_e201_risk")
        ].shape[0]
        == 4
    )
    rows = [
        {
            "gate": "input_integrity",
            "passed": input_integrity,
            "observed": (
                f"pretruth_max={pretruth_max:.8g};"
                f"family_mean_max={risk_status['verified_family_mean_max_abs_residual']:.8g};"
                "official_general_equivalence="
                f"{general_status['e200_official_code_equivalence']['maximum_absolute_delta_residual']:.8g}"
            ),
            "criterion": "2,008 aligned tasks; pretruth residual <5e-6",
            "gate_type": "execution",
        },
        {
            "gate": "family_error_certificate",
            "passed": certificate,
            "observed": (
                f"identity_max={identity_max:.8g};"
                f"lower_bound_violations={lower_violations}"
            ),
            "criterion": "identity residual <=1e-10; zero lower-bound violations",
            "gate_type": "deterministic_certificate",
        },
        {
            "gate": "empirical_routing",
            "passed": routing,
            "observed": (
                f"rho_lower={pooled_assoc.ci95_lower:.8g};"
                f"utility_lower={pooled_utility.utility_ci95_lower:.8g}"
            ),
            "criterion": "both pooled 95% cluster-bootstrap lower bounds >0",
            "gate_type": "scientific",
        },
        {
            "gate": "incremental_vs_magnitude",
            "passed": incremental,
            "observed": (
                f"partial_lower={pooled_partial.ci95_lower:.8g};"
                f"delta_utility_lower={pooled_increment.ci95_lower:.8g}"
            ),
            "criterion": "either paired 95% cluster-bootstrap lower bound >0",
            "gate_type": "scientific",
        },
        {
            "gate": "target_reporting_complete",
            "passed": target_reporting,
            "observed": "K562,RPE1,hepg2,jurkat",
            "criterion": "all four targets reported without directional filtering",
            "gate_type": "reporting",
        },
    ]
    statuses = {
        "execution": "PASS" if input_integrity and target_reporting else "FAIL",
        "certificate": "PASS" if certificate else "FAIL",
        "empirical_routing": "SUPPORTED" if routing else "NOT_SUPPORTED",
        "incremental_vs_magnitude": ("SUPPORTED" if incremental else "NOT_SUPPORTED"),
    }
    return pd.DataFrame(rows), statuses


def write_report(
    task_frame: pd.DataFrame,
    associations: pd.DataFrame,
    partials: pd.DataFrame,
    utilities: pd.DataFrame,
    increments: pd.DataFrame,
    baselines: pd.DataFrame,
    target_summary: pd.DataFrame,
    gates: pd.DataFrame,
    statuses: dict[str, str],
) -> None:
    pooled_assoc = associations.loc[associations.scope.eq("pooled")][
        ["predictor", "estimate", "ci95_lower", "ci95_upper"]
    ]
    pooled_partial = partials.loc[partials.scope.eq("pooled")][
        ["predictor", "covariate", "estimate", "ci95_lower", "ci95_upper"]
    ]
    pooled_utility = utilities.loc[utilities.scope.eq("pooled")][
        [
            "predictor",
            "n_selected",
            "high_error_capture",
            "error_lift",
            "oracle_normalized_utility",
            "utility_ci95_lower",
            "utility_ci95_upper",
        ]
    ]
    pooled_increment = increments.loc[increments.scope.eq("pooled")][
        ["measure", "estimate", "ci95_lower", "ci95_upper"]
    ]
    pooled_baselines = baselines.loc[baselines.scope.eq("pooled")][
        [
            "baseline",
            "predictor_mean_error",
            "baseline_mean_error",
            "task_win_rate",
            "estimate",
            "ci95_lower",
            "ci95_upper",
        ]
    ].rename(columns={"estimate": "mean_delta"})
    target_assoc = associations.loc[
        associations.scope.isin(TARGETS)
        & associations.predictor.eq("safeconf_e201_risk")
    ][["scope", "n_tasks", "estimate", "ci95_lower", "ci95_upper"]]
    primary_target_summary = target_summary.loc[
        target_summary.stratum.eq("primary_ge30")
        & target_summary.predictor.eq("four_seed_family")
    ][
        [
            "target",
            "n_tasks",
            "family_centroid_rmse_mean",
            "family_rms_error_mean",
            "worst_seed_error_mean",
            "control_error_mean",
            "official_general_baseline_error_mean",
            "source_transfer_error_mean",
        ]
    ]
    identity_max = task_frame.family_rms_sq_identity_residual.abs().max()
    lower_violations = int(task_frame.family_rms_lower_bound_violation.sum())
    lines = [
        "# E201 四目标四种子核心评价",
        "",
        f"- 执行完整性：**{statuses['execution']}**。",
        f"- family-error 代数证书：**{statuses['certificate']}**。",
        f"- 预测前经验路由：**{statuses['empirical_routing']}**。",
        f"- 相对 predicted magnitude 的增量：**{statuses['incremental_vs_magnitude']}**。",
        "",
        "## 证书复核",
        "",
        f"1,808 个主任务和 200 个敏感性任务均已重算。恒等式最大绝对残差为 "
        f"`{identity_max:.8g}`，family RMS 低于 disagreement 的任务数为 "
        f"`{lower_violations}`。证书只说明四种子分歧是 family RMS error 的确定性"
        "下界，不替代经验排序结果。",
        "",
        "## 主关联",
        "",
        markdown_table(pooled_assoc),
        "",
        "## 控制 predicted magnitude",
        "",
        markdown_table(pooled_partial),
        "",
        "## 固定 20% 复核预算",
        "",
        markdown_table(pooled_utility),
        "",
        "## SafeConf 相对 magnitude 的配对增量",
        "",
        markdown_table(pooled_increment),
        "",
        "## family centroid 与简单基线",
        "",
        markdown_table(pooled_baselines),
        "",
        "## 四个 target",
        "",
        markdown_table(target_assoc),
        "",
        markdown_table(primary_target_summary),
        "",
        "## 正式门",
        "",
        markdown_table(gates),
        "",
        "## 解释边界",
        "",
        "E201 检查公开 TxPert STRING-GAT 在四个跨细胞背景目标上的四种子重训练"
        "family。任何未通过的科学门均保留为 NOT_SUPPORTED；不会因 pooled 或某个"
        "target 的方向不利而删除任务，也不会把代数证书写成对所有模型、扰动模态"
        "或数据集的经验有效性。scPertEval 五端点在独立补充程序中运行。",
        "",
    ]
    atomic_text(REPORTS / "E201_CORE_REPORT.md", "\n".join(lines))


def self_test() -> None:
    rng = np.random.default_rng(20260802)
    n_tasks, n_genes, n_seeds = 48, 19, 4
    truth = rng.normal(size=(n_tasks, n_genes))
    seeds = truth[None, :, :] + rng.normal(scale=0.3, size=(n_seeds, n_tasks, n_genes))
    family = seeds.mean(axis=0)
    seed_errors = np.sqrt(np.mean((seeds - truth[None, :, :]) ** 2, axis=2))
    family_rms = np.sqrt(np.mean(seed_errors**2, axis=0))
    centroid_error = np.sqrt(np.mean((family - truth) ** 2, axis=1))
    disagreement = np.sqrt(np.mean((seeds - family[None, :, :]) ** 2, axis=(0, 2)))
    identity = family_rms**2 - centroid_error**2 - disagreement**2
    if np.max(np.abs(identity)) > IDENTITY_TOL:
        raise EvaluationFailure("synthetic family-error identity failed")
    difficulty = rng.normal(size=n_tasks)
    magnitude = rng.normal(size=n_tasks)
    outcome = 0.8 * difficulty + 0.8 * magnitude + rng.normal(scale=0.1, size=n_tasks)
    risk = difficulty + rng.normal(scale=0.1, size=n_tasks)
    if partial_spearman(risk, outcome, magnitude) <= 0:
        raise EvaluationFailure("synthetic partial Spearman failed")
    frame = pd.DataFrame(
        {
            "task_id": [f"T{i % 2}::{i}" for i in range(n_tasks)],
            "target": ["K562" if i % 2 == 0 else "RPE1" for i in range(n_tasks)],
            "condition": [f"G{i // 2}+ctrl" for i in range(n_tasks)],
            "safeconf_e201_risk": risk,
            "predicted_magnitude": magnitude,
            "family_disagreement": disagreement,
            "family_rms_error": outcome - outcome.min() + 0.1,
            "family_centroid_rmse": centroid_error,
            "control_error": centroid_error + 0.1,
            "official_general_baseline_error": centroid_error + 0.08,
            "source_transfer_error": centroid_error + 0.05,
        }
    )
    point, draws = bootstrap_scope(frame, "synthetic", n_bootstrap=100)
    if len(draws) != 100 or not all(math.isfinite(value) for value in point.values()):
        raise EvaluationFailure("synthetic cluster bootstrap failed")
    result = {
        "status": "PASS",
        "family_identity_max_abs_residual": float(np.max(np.abs(identity))),
        "partial_spearman_positive": True,
        "cluster_bootstrap_draws": len(draws),
        "target_truth_opened": False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    data_root = args.data_root.resolve()
    vector_dir = args.evaluation_vector_dir.resolve()
    try:
        vector_dir.relative_to(data_root)
    except ValueError as exc:
        raise EvaluationFailure(
            "evaluation vector output must stay under data root"
        ) from exc
    if FINAL.exists() or vector_dir.exists():
        raise EvaluationFailure(
            "formal core evaluation is append-only and already exists"
        )
    head = verify_git_release()
    features, vectors, risk_status, risk_inputs = verify_risk_inputs(data_root)
    general_centroids, general_status, general_inputs = verify_general_baseline(
        data_root, risk_status
    )
    truth_release, release_status, truth_inputs = verify_truth_release(
        data_root, risk_status, general_status
    )
    task_metrics, seed_metrics, truth_centroids = compute_task_metrics(
        features, vectors, general_centroids, truth_release
    )
    task_frame = features.merge(
        task_metrics,
        on=[
            "task_id",
            "target",
            "condition",
            "gene",
            "analysis_stratum",
            "n_target_cells",
        ],
        validate="one_to_one",
    )
    primary = task_frame.loc[task_frame.analysis_stratum.eq("primary_ge30")].copy()
    sensitivity = task_frame.loc[
        task_frame.analysis_stratum.eq("sensitivity_10_29")
    ].copy()
    if len(primary) != N_PRIMARY or len(sensitivity) != N_SENSITIVITY:
        raise EvaluationFailure("formal analysis strata changed")

    (
        associations,
        partials,
        utilities,
        increments,
        baselines,
        pooled_draws,
    ) = build_inference_tables(primary)
    descriptive_primary = descriptive_associations(primary)
    sensitivity_associations = descriptive_associations(sensitivity)
    target_summary = target_error_summary(task_metrics, seed_metrics)
    gates, statuses = build_gates(
        task_frame,
        risk_status,
        general_status,
        associations,
        partials,
        utilities,
        increments,
    )
    if statuses["execution"] != "PASS":
        raise EvaluationFailure("formal execution-integrity gate failed")

    for directory in (TABLES, REPORTS, FIGURES):
        directory.mkdir(parents=True, exist_ok=True)
    input_hashes = pd.DataFrame(risk_inputs + general_inputs + truth_inputs)
    atomic_csv(TABLES / "E201_CORE_INPUT_HASHES.csv", input_hashes)
    atomic_csv(TABLES / "E201_TASK_METRICS.csv", task_frame)
    atomic_csv(TABLES / "E201_SEED_TASK_ERRORS.csv", seed_metrics)
    atomic_csv(TABLES / "E201_TARGET_ERROR_SUMMARY.csv", target_summary)
    atomic_csv(TABLES / "E201_RISK_ASSOCIATIONS.csv", associations)
    atomic_csv(TABLES / "E201_PARTIAL_ASSOCIATIONS.csv", partials)
    atomic_csv(TABLES / "E201_REVIEW_UTILITY.csv", utilities)
    atomic_csv(TABLES / "E201_INCREMENTAL_TESTS.csv", increments)
    atomic_csv(TABLES / "E201_BASELINE_COMPARISONS.csv", baselines)
    atomic_csv(TABLES / "E201_DESCRIPTIVE_ASSOCIATIONS.csv", descriptive_primary)
    atomic_csv(
        TABLES / "E201_SENSITIVITY_ASSOCIATIONS.csv",
        sensitivity_associations,
    )
    atomic_csv(TABLES / "E201_MAIN_CLUSTER_BOOTSTRAP_DRAWS.csv", pooled_draws)
    atomic_csv(TABLES / "E201_FORMAL_GATES.csv", gates)
    write_report(
        task_frame,
        associations,
        partials,
        utilities,
        increments,
        baselines,
        target_summary,
        gates,
        statuses,
    )
    make_figure(
        primary,
        associations,
        utilities,
        FIGURES / "E201_core_audit.png",
    )

    vector_dir.mkdir(parents=True)
    truth_centroid_path = vector_dir / "E201_TARGET_TRUTH_CENTROIDS.npy"
    atomic_npy(truth_centroid_path, truth_centroids)
    truth_centroid_record = {
        **file_record(truth_centroid_path, data_root),
        "shape": list(truth_centroids.shape),
        "dtype": str(truth_centroids.dtype),
    }

    output_rows = []
    for path in sorted(
        candidate for candidate in FINAL.rglob("*") if candidate.is_file()
    ):
        output_rows.append(file_record(path, data_root))
    output_hashes = pd.DataFrame(output_rows)
    output_hash_path = TABLES / "E201_CORE_OUTPUT_HASHES.csv"
    atomic_csv(output_hash_path, output_hashes)
    status = {
        "experiment": "E201_txpert_multitarget_retraining",
        "stage": "FORMAL_CORE_EVALUATION",
        "status": "PASS",
        "generated_at": now(),
        "git_head": head,
        "pretruth_risk_status_sha256": sha256_file(RISK_STATUS),
        "pretruth_official_general_baseline_status_sha256": sha256_file(GENERAL_STATUS),
        "target_truth_release_status_sha256": sha256_file(RELEASE_STATUS),
        "target_truth_released_after_risk_seal": bool(
            release_status["pretruth_risk_sealed_before_release"]
        ),
        "target_truth_released_after_general_baseline_seal": bool(
            release_status["pretruth_official_general_baseline_sealed_before_release"]
        ),
        "n_tasks": N_TASKS,
        "n_primary_tasks": N_PRIMARY,
        "n_sensitivity_tasks": N_SENSITIVITY,
        "n_bootstrap": N_BOOTSTRAP,
        "bootstrap_unit": "perturbation_condition_cluster_across_targets",
        "review_budget": BUDGET,
        "execution_status": statuses["execution"],
        "certificate_status": statuses["certificate"],
        "empirical_routing_status": statuses["empirical_routing"],
        "incremental_vs_magnitude_status": statuses["incremental_vs_magnitude"],
        "family_identity_max_abs_residual": float(
            task_frame.family_rms_sq_identity_residual.abs().max()
        ),
        "family_lower_bound_violations": int(
            task_frame.family_rms_lower_bound_violation.sum()
        ),
        "truth_centroid_file": truth_centroid_record,
        "output_hash_table": file_record(output_hash_path, data_root),
    }
    atomic_json(FINAL_STATUS, status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
