#!/usr/bin/env python3
"""Build and seal E205 risk features without opening target outcomes.

The runner accepts only the sealed Exphormer predictions, matched controls and
the source-only E201 task base.  It deliberately has no truth-path argument.
Large task-centroid arrays are written to DATA while the small risk table and
status record can be committed to the repository before evaluation is
authorized.
"""

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
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
E201 = ROOT / "docs/实验结果/E201_txpert_multitarget_retraining_20260802"
TASK_STATUS = E201 / "E201_PRETRUTH_TASK_BASE_STATUS.json"
TASK_TABLE = E201 / "tables/E201_PRETRUTH_TASK_BASE.csv"
GAT_RISK_STATUS = E201 / "E201_PRETRUTH_RISK_STATUS.json"
TARGETS = ("K562", "RPE1", "hepg2", "jurkat")
SEEDS = (1, 2, 3, 4)
EXPECTED_SAMPLES = {
    "K562": 150_472,
    "RPE1": 67_034,
    "hepg2": 54_911,
    "jurkat": 81_791,
}
N_TASKS = 2_008
N_PRIMARY = 1_808
N_GENES = 3_352
RISK_COMPONENTS = (
    "family_disagreement",
    "model_source_gap",
    "source_delta_dispersion",
    "negative_log_source_cells",
    "support_context_deficit",
)
TAU_QUANTILES = tuple(number / 10 for number in range(1, 10))


class RiskFailure(RuntimeError):
    """Fail-closed E205 pretruth risk error."""


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--family-seal", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--risk-table", type=Path, required=True)
    parser.add_argument("--risk-status", type=Path, required=True)
    parser.add_argument("--vector-output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_data_path(value: str, data_root: Path) -> Path:
    if not value.startswith("DATA/"):
        raise RiskFailure(f"not a DATA-relative path: {value}")
    result = (data_root / value[len("DATA/") :]).resolve()
    try:
        result.relative_to(data_root.resolve())
    except ValueError as exc:
        raise RiskFailure(f"DATA path escapes root: {value}") from exc
    return result


def data_path(path: Path, data_root: Path) -> str:
    try:
        relative = path.resolve().relative_to(data_root.resolve())
    except ValueError as exc:
        raise RiskFailure(f"vector is outside data root: {path}") from exc
    return "DATA/" + relative.as_posix()


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


def tracked_clean(path: Path) -> bool:
    try:
        relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return False
    commands = (
        ["git", "-C", str(ROOT), "cat-file", "-e", f"HEAD:{relative}"],
        ["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", relative],
        ["git", "-C", str(ROOT), "diff", "--cached", "--quiet", "HEAD", "--", relative],
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


def git_text(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args], text=True
    ).strip()


def verify_git_release(family_seal: Path) -> str:
    required = (SCRIPT, TASK_STATUS, TASK_TABLE, GAT_RISK_STATUS, family_seal)
    if not all(path.is_file() and tracked_clean(path) for path in required):
        raise RiskFailure("risk code and all sealed inputs must be tracked and clean")
    branch = git_text("branch", "--show-current")
    head = git_text("rev-parse", "HEAD")
    if not branch:
        raise RiskFailure("detached HEAD is not allowed")
    for remote in ("origin", "github"):
        if git_text("rev-parse", f"{remote}/{branch}") != head:
            raise RiskFailure(f"{remote}/{branch} differs from local HEAD")
    return head


def ensure_no_truth_artifacts(prediction_root: Path) -> None:
    if not prediction_root.is_dir():
        raise RiskFailure(f"missing prediction root: {prediction_root}")
    forbidden = [
        path
        for path in prediction_root.rglob("*")
        if "truth" in path.name.casefold()
    ]
    if forbidden:
        raise RiskFailure(f"truth artifact exists under prediction root: {forbidden[:3]}")


def rmse(left: np.ndarray, right: np.ndarray) -> float:
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(delta))))


def condition_from_label(target: str, label: str) -> str:
    prefix, suffix = f"{target}_", "_1+1"
    if not label.startswith(prefix) or not label.endswith(suffix):
        raise RiskFailure(f"unexpected target condition label: {label}")
    condition = label[len(prefix) : -len(suffix)]
    if not condition.endswith("+ctrl"):
        raise RiskFailure(f"not a single-perturbation label: {label}")
    return condition


def standardize_with_primary(frame: pd.DataFrame, column: str) -> pd.Series:
    primary = frame.loc[frame.analysis_stratum.eq("primary_ge30"), column].astype(float)
    center, scale = float(primary.mean()), float(primary.std(ddof=0))
    if not math.isfinite(center) or not math.isfinite(scale) or scale <= 0:
        raise RiskFailure(f"invalid within-target standardization: {column}")
    return (frame[column].astype(float) - center) / scale


def rank_fusion_4_to_1(magnitude: np.ndarray, safeconf: np.ndarray) -> np.ndarray:
    magnitude = np.asarray(magnitude, dtype=float)
    safeconf = np.asarray(safeconf, dtype=float)
    if (
        magnitude.shape != safeconf.shape
        or magnitude.ndim != 1
        or len(magnitude) < 2
        or not np.isfinite(magnitude).all()
        or not np.isfinite(safeconf).all()
    ):
        raise RiskFailure("4:1 rank fusion requires aligned finite vectors")
    m_rank = rankdata(magnitude, method="average")
    s_rank = rankdata(safeconf, method="average")
    return (4.0 * m_rank + s_rank) / (5.0 * len(magnitude))


def architecture_aware_router(
    magnitude: np.ndarray,
    safeconf: np.ndarray,
    cross_architecture_disagreement: np.ndarray,
) -> np.ndarray:
    """Frozen E217 cross-architecture monotone correction.

    E217 selected the two coefficients on released E153 data before E205
    target truth was authorized.  Every input is converted to a within-target
    percentile.  Evidence may only raise the magnitude-anchored score.
    """
    arrays = [
        np.asarray(magnitude, dtype=float),
        np.asarray(safeconf, dtype=float),
        np.asarray(cross_architecture_disagreement, dtype=float),
    ]
    if (
        any(value.ndim != 1 for value in arrays)
        or len({value.shape for value in arrays}) != 1
        or len(arrays[0]) < 2
        or any(not np.isfinite(value).all() for value in arrays)
    ):
        raise RiskFailure("architecture-aware router requires aligned finite vectors")
    n_tasks = len(arrays[0])
    magnitude_rank, safeconf_rank, disagreement_rank = [
        rankdata(value, method="average") / n_tasks for value in arrays
    ]
    return (
        magnitude_rank
        + 0.50 * np.maximum(safeconf_rank - magnitude_rank, 0.0)
        + 0.125 * np.maximum(disagreement_rank - magnitude_rank, 0.0)
    )


def certificate_priority_score(
    magnitude: np.ndarray,
    lower_bound: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Prioritize frozen high-error certificates, then break ties by magnitude.

    The score is deliberately lexicographic rather than a tuned weighted sum:
    every task with ``lower_bound > tau`` ranks above every uncertified task,
    while predicted magnitude determines order within the two groups.
    """
    magnitude = np.asarray(magnitude, dtype=float)
    lower_bound = np.asarray(lower_bound, dtype=float)
    if (
        magnitude.shape != lower_bound.shape
        or magnitude.ndim != 1
        or len(magnitude) < 2
        or not np.isfinite(magnitude).all()
        or not np.isfinite(lower_bound).all()
        or not math.isfinite(float(tau))
    ):
        raise RiskFailure("certificate-priority routing requires aligned finite inputs")
    magnitude_rank = rankdata(magnitude, method="average") / len(magnitude)
    certified = (lower_bound > float(tau)).astype(float)
    return 2.0 * certified + magnitude_rank


def verify_record(path: Path, record: dict, label: str) -> None:
    if (
        not path.is_file()
        or path.stat().st_size != int(record.get("bytes", -1))
        or sha256_file(path) != record.get("sha256")
    ):
        raise RiskFailure(f"sealed {label} changed: {path}")


def load_base_inputs(data_root: Path) -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray]]:
    task_status = json.loads(TASK_STATUS.read_text(encoding="utf-8"))
    if (
        task_status.get("status") != "PASS"
        or int(task_status.get("n_tasks", -1)) != N_TASKS
        or int(task_status.get("n_primary_tasks", -1)) != N_PRIMARY
        or int(task_status.get("target_perturbed_expression_rows_opened", -1)) != 0
        or task_status.get("target_outcomes_evaluated") is not False
    ):
        raise RiskFailure("E201 source-only task base failed")
    for record in task_status.get("tracked_outputs", []):
        verify_record(ROOT / record["path"], record, "task-base file")
    source_record = task_status.get("source_mean_delta_file", {})
    source_path = resolve_data_path(source_record.get("path", ""), data_root)
    verify_record(source_path, source_record, "source mean delta")
    tasks = pd.read_csv(TASK_TABLE, keep_default_na=True)
    source_mean = np.load(source_path, mmap_mode="r", allow_pickle=False)
    if (
        len(tasks) != N_TASKS
        or tasks.task_id.nunique() != N_TASKS
        or source_mean.shape != (N_TASKS, N_GENES)
        or not np.array_equal(tasks.source_mean_delta_row.to_numpy(), np.arange(N_TASKS))
    ):
        raise RiskFailure("source-only task alignment changed")

    gat_status = json.loads(GAT_RISK_STATUS.read_text(encoding="utf-8"))
    if (
        gat_status.get("status") != "PASS"
        or int(gat_status.get("n_tasks", -1)) != N_TASKS
        or gat_status.get("target_truth_materialized") is not False
        or gat_status.get("target_outcomes_evaluated") is not False
    ):
        raise RiskFailure("E201 pretruth GAT vectors failed")
    records = {Path(item["path"]).name: item for item in gat_status.get("vector_files", [])}
    names = (
        "E201_SEED_CENTROIDS.npy",
        "E201_FAMILY_CENTROIDS.npy",
        "E201_CONTROL_CENTROIDS.npy",
    )
    if not set(names).issubset(records):
        raise RiskFailure("E201 pretruth GAT centroid records are incomplete")
    gat_vectors = {}
    for name in names:
        path = resolve_data_path(records[name]["path"], data_root)
        verify_record(path, records[name], name)
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        expected_shape = (
            (4, N_TASKS, N_GENES)
            if name == "E201_SEED_CENTROIDS.npy"
            else (N_TASKS, N_GENES)
        )
        if values.shape != expected_shape or str(values.dtype) != "float32":
            raise RiskFailure(f"unexpected GAT vector contract: {name}")
        gat_vectors[name] = values
    return tasks, source_mean, gat_vectors


def verify_family_seal(path: Path) -> tuple[dict, str]:
    seal = json.loads(path.read_text(encoding="utf-8"))
    records = seal.get("records", [])
    jobs = {(row.get("target"), int(row.get("seed", -1))) for row in records}
    expected = {(target, seed) for target in TARGETS for seed in SEEDS}
    if (
        seal.get("experiment") != "E205_cross_family_exphormer"
        or seal.get("status") != "SEALED_16_EXPHORMER_CHECKPOINTS"
        or seal.get("model_family") != "TxPert-Exphormer"
        or seal.get("target_truth_opened") is not False
        or int(seal.get("n_jobs", -1)) != 16
        or jobs != expected
        or canonical_hash(records) != seal.get("records_sha256")
    ):
        raise RiskFailure("E205 checkpoint family seal failed")
    return seal, sha256_file(path)


def load_target_predictions(
    target: str, prediction_root: Path, family_seal_sha: str
) -> tuple[pd.DataFrame, np.ndarray, list[np.ndarray], list[dict]]:
    target_root = prediction_root / target
    shared = target_root / "shared"
    manifest_path = shared / "E205_SHARED_TARGET_MANIFEST.json"
    if not manifest_path.is_file():
        raise RiskFailure(f"missing E205 shared manifest: {target}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("experiment") != "E205_cross_family_exphormer"
        or manifest.get("status") != "SEALED_PRETRUTH_TARGET_INPUTS"
        or manifest.get("target") != target
        or int(manifest.get("n_samples", -1)) != EXPECTED_SAMPLES[target]
        or int(manifest.get("n_genes", -1)) != N_GENES
        or manifest.get("target_truth_materialized") is not False
        or int(manifest.get("target_expression_nonzero_values_seen", -1)) != 0
    ):
        raise RiskFailure(f"E205 shared pretruth manifest failed: {target}")
    entries = {item.get("role"): item for item in manifest.get("files", [])}
    if set(entries) != {"controls", "observations"}:
        raise RiskFailure(f"unexpected E205 shared roles: {target}")
    controls_path, observations_path = shared / "controls.npy", shared / "observations.csv"
    verify_record(controls_path, entries["controls"], f"{target} controls")
    verify_record(observations_path, entries["observations"], f"{target} observations")
    controls = np.load(controls_path, mmap_mode="r", allow_pickle=False)
    observations = pd.read_csv(observations_path, keep_default_na=False)
    if (
        controls.shape != (EXPECTED_SAMPLES[target], N_GENES)
        or len(observations) != EXPECTED_SAMPLES[target]
        or observations.row_index.tolist() != list(range(len(observations)))
        or set(observations.cell_type.astype(str)) != {target}
    ):
        raise RiskFailure(f"E205 shared target alignment changed: {target}")

    predictions, statuses = [], []
    shared_sha = sha256_file(manifest_path)
    for seed in SEEDS:
        run_dir = target_root / f"seed_{seed}"
        status_path = run_dir / "E205_PREDICTION_RUN.json"
        if not status_path.is_file():
            raise RiskFailure(f"missing E205 prediction status: {target}/seed_{seed}")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if (
            status.get("experiment") != "E205_cross_family_exphormer"
            or status.get("status") != "COMPLETE"
            or status.get("model_family") != "TxPert-Exphormer"
            or status.get("architecture") != "exphormer"
            or status.get("target") != target
            or int(status.get("seed", -1)) != seed
            or status.get("checkpoint_role") != "last"
            or status.get("family_seal_sha256") != family_seal_sha
            or status.get("target_truth_materialized") is not False
            or int(status.get("target_expression_nonzero_values_seen", -1)) != 0
            or status.get("shared_target_manifest_sha256") != shared_sha
        ):
            raise RiskFailure(f"E205 prediction status failed: {target}/seed_{seed}")
        record = status.get("prediction_file", {})
        path = run_dir / record.get("path", "")
        verify_record(path, record, f"{target}/seed_{seed} prediction")
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.shape != (EXPECTED_SAMPLES[target], N_GENES):
            raise RiskFailure(f"E205 prediction shape changed: {target}/seed_{seed}")
        predictions.append(values)
        statuses.append({"path": status_path, "status": status})
    return observations, controls, predictions, statuses


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    family_seal_path = args.family_seal.resolve()
    prediction_root = args.prediction_root.resolve()
    risk_table = args.risk_table.resolve()
    risk_status = args.risk_status.resolve()
    vector_dir = args.vector_output_dir.resolve()
    if risk_table.exists() or risk_status.exists() or vector_dir.exists():
        raise RiskFailure("refusing to overwrite pretruth E205 output")
    ensure_no_truth_artifacts(prediction_root)
    safeconf_commit = verify_git_release(family_seal_path)
    tasks, source_mean, gat = load_base_inputs(data_root)
    _, family_seal_sha = verify_family_seal(family_seal_path)

    feature_blocks = []
    seed_blocks, family_blocks, registered_blocks = [], [], []
    control_blocks, source_blocks = [], []
    prediction_inputs = []
    for target in TARGETS:
        target_tasks = tasks.loc[tasks.target.eq(target)].copy()
        observations, controls, predictions, statuses = load_target_predictions(
            target, prediction_root, family_seal_sha
        )
        prediction_inputs.extend(
            {
                "target": target,
                "seed": int(item["status"]["seed"]),
                "status_sha256": sha256_file(item["path"]),
                "prediction_sha256": item["status"]["prediction_file"]["sha256"],
            }
            for item in statuses
        )
        labels = observations.pert_cond_name.astype(str).map(
            lambda value: condition_from_label(target, value)
        ).to_numpy()
        rows, target_seed, target_family, target_registered = [], [], [], []
        target_control, target_source = [], []
        for task in target_tasks.itertuples(index=False):
            indices = np.flatnonzero(labels == str(task.condition))
            if len(indices) != int(task.n_target_cells):
                raise RiskFailure(f"target cell count changed: {task.task_id}")
            seed_centroids = np.stack(
                [
                    np.asarray(values[indices], dtype=np.float64).mean(axis=0)
                    for values in predictions
                ]
            )
            family = seed_centroids.mean(axis=0)
            control = np.asarray(controls[indices], dtype=np.float64).mean(axis=0)
            source_delta = np.asarray(
                source_mean[int(task.source_mean_delta_row)], dtype=np.float64
            )
            source_prediction = control + source_delta
            gat_family = np.asarray(
                gat["E201_FAMILY_CENTROIDS.npy"][int(task.source_mean_delta_row)],
                dtype=np.float64,
            )
            gat_seed_centroids = np.asarray(
                gat["E201_SEED_CENTROIDS.npy"][:, int(task.source_mean_delta_row)],
                dtype=np.float64,
            )
            gat_control = np.asarray(
                gat["E201_CONTROL_CENTROIDS.npy"][int(task.source_mean_delta_row)],
                dtype=np.float64,
            )
            registered_members = np.concatenate(
                [gat_seed_centroids, seed_centroids], axis=0
            )
            registered_family = registered_members.mean(axis=0)
            exphormer_disagreement = rmse(seed_centroids, family[None, :])
            gat_disagreement = rmse(gat_seed_centroids, gat_family[None, :])
            registered_disagreement = rmse(
                registered_members, registered_family[None, :]
            )
            rows.append(
                {
                    "task_id": task.task_id,
                    # Keep the historical name for the frozen empirical-risk
                    # formula; it refers to the Exphormer seed family.
                    "family_disagreement": exphormer_disagreement,
                    "exphormer_family_disagreement": exphormer_disagreement,
                    "gat_family_disagreement": gat_disagreement,
                    "registered_family_disagreement": registered_disagreement,
                    "family_radius": float(
                        np.max(
                            np.sqrt(
                                np.mean(
                                    np.square(seed_centroids - family[None, :]), axis=1
                                )
                            )
                        )
                    ),
                    "registered_family_radius": float(
                        np.max(
                            np.sqrt(
                                np.mean(
                                    np.square(
                                        registered_members
                                        - registered_family[None, :]
                                    ),
                                    axis=1,
                                )
                            )
                        )
                    ),
                    "predicted_magnitude": rmse(family, control),
                    "registered_predicted_magnitude": rmse(
                        registered_family, control
                    ),
                    "model_source_gap": rmse(family, source_prediction),
                    "source_transfer_magnitude": rmse(source_prediction, control),
                    "cross_family_disagreement": rmse(family, gat_family),
                    "gat_control_residual": rmse(control, gat_control),
                }
            )
            target_seed.append(seed_centroids.astype(np.float32))
            target_family.append(family.astype(np.float32))
            target_registered.append(registered_family.astype(np.float32))
            target_control.append(control.astype(np.float32))
            target_source.append(source_prediction.astype(np.float32))
        block = target_tasks.merge(
            pd.DataFrame(rows), on="task_id", how="left", validate="one_to_one"
        )
        for component in RISK_COMPONENTS:
            block[f"z_{component}"] = standardize_with_primary(block, component)
        block["safeconf_e205_risk"] = block[
            [f"z_{name}" for name in RISK_COMPONENTS]
        ].mean(axis=1)
        block["safeconf_m_4to1"] = rank_fusion_4_to_1(
            block.predicted_magnitude.to_numpy(float),
            block.safeconf_e205_risk.to_numpy(float),
        )
        block["architecture_aware_router"] = architecture_aware_router(
            block.registered_predicted_magnitude.to_numpy(float),
            block.safeconf_e205_risk.to_numpy(float),
            block.cross_family_disagreement.to_numpy(float),
        )
        feature_blocks.append(block)
        seed_blocks.append(np.stack(target_seed, axis=1))
        family_blocks.append(np.stack(target_family))
        registered_blocks.append(np.stack(target_registered))
        control_blocks.append(np.stack(target_control))
        source_blocks.append(np.stack(target_source))

    features = pd.concat(feature_blocks, ignore_index=True)
    arrays = {
        "E205_SEED_CENTROIDS.npy": np.concatenate(seed_blocks, axis=1),
        "E205_FAMILY_CENTROIDS.npy": np.concatenate(family_blocks),
        "E205_REGISTERED_FAMILY_CENTROIDS.npy": np.concatenate(registered_blocks),
        "E205_CONTROL_CENTROIDS.npy": np.concatenate(control_blocks),
        "E205_SOURCE_TRANSFER_CENTROIDS.npy": np.concatenate(source_blocks),
    }
    if (
        len(features) != N_TASKS
        or features.task_id.nunique() != N_TASKS
        or int(features.analysis_stratum.eq("primary_ge30").sum()) != N_PRIMARY
        or not np.array_equal(
            features.source_mean_delta_row.to_numpy(int), np.arange(N_TASKS)
        )
        or arrays["E205_SEED_CENTROIDS.npy"].shape != (4, N_TASKS, N_GENES)
        or arrays["E205_REGISTERED_FAMILY_CENTROIDS.npy"].shape
        != (N_TASKS, N_GENES)
        or any(not np.isfinite(values).all() for values in arrays.values())
        or not np.isfinite(
            features[
                [
                    "safeconf_e205_risk",
                    "safeconf_m_4to1",
                    "predicted_magnitude",
                    "cross_family_disagreement",
                    "registered_family_disagreement",
                    "gat_family_disagreement",
                    "architecture_aware_router",
                ]
            ].to_numpy(float)
        ).all()
    ):
        raise RiskFailure("combined E205 pretruth feature contract failed")
    family_residual = float(
        np.max(
            np.abs(
                arrays["E205_SEED_CENTROIDS.npy"].mean(axis=0)
                - arrays["E205_FAMILY_CENTROIDS.npy"]
            )
        )
    )
    registered_centroid_residual = float(
        np.max(
            np.abs(
                (
                    gat["E201_FAMILY_CENTROIDS.npy"].astype(np.float64)
                    + arrays["E205_FAMILY_CENTROIDS.npy"].astype(np.float64)
                )
                / 2.0
                - arrays["E205_REGISTERED_FAMILY_CENTROIDS.npy"].astype(
                    np.float64
                )
            )
        )
    )
    control_residual = float(features.gat_control_residual.max())
    if (
        family_residual > 2e-6
        or registered_centroid_residual > 2e-6
        or control_residual > 2e-6
    ):
        raise RiskFailure(
            "centroid alignment failed: "
            f"family={family_residual}, "
            f"registered={registered_centroid_residual}, "
            f"control={control_residual}"
        )

    primary_disagreement = features.loc[
        features.analysis_stratum.eq("primary_ge30"),
        "registered_family_disagreement",
    ].to_numpy(float)
    tau_grid = [
        {
            "quantile": quantile,
            "tau": float(np.quantile(primary_disagreement, quantile)),
        }
        for quantile in TAU_QUANTILES
    ]
    tau_q80 = next(
        float(record["tau"])
        for record in tau_grid
        if math.isclose(float(record["quantile"]), 0.8)
    )
    features["certificate_priority_q80"] = np.nan
    for target in TARGETS:
        index = features.index[features.target.eq(target)]
        features.loc[index, "certificate_priority_q80"] = certificate_priority_score(
            features.loc[index, "registered_predicted_magnitude"].to_numpy(float),
            features.loc[index, "registered_family_disagreement"].to_numpy(float),
            tau_q80,
        )
    if not np.isfinite(
        features[
            [
                "registered_predicted_magnitude",
                "certificate_priority_q80",
            ]
        ].to_numpy(float)
    ).all():
        raise RiskFailure("certificate-priority pretruth score is non-finite")

    vector_dir.mkdir(parents=True)
    vector_records = []
    for filename, values in arrays.items():
        path = vector_dir / filename
        atomic_npy(path, values.astype(np.float32, copy=False))
        vector_records.append(
            {
                "path": data_path(path, data_root),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "shape": list(values.shape),
                "dtype": "float32",
            }
        )
    atomic_csv(risk_table, features)
    risk_record = {
        "path": risk_table.relative_to(ROOT).as_posix(),
        "bytes": risk_table.stat().st_size,
        "sha256": sha256_file(risk_table),
    }
    status = {
        "experiment": "E205_cross_family_exphormer",
        "stage": "PRETRUTH_RISK_FEATURES",
        "status": "PASS",
        "generated_at": now(),
        "safeconf_commit": safeconf_commit,
        "family_seal_sha256": family_seal_sha,
        "task_base_status_sha256": sha256_file(TASK_STATUS),
        "gat_pretruth_risk_status_sha256": sha256_file(GAT_RISK_STATUS),
        "n_tasks": len(features),
        "n_primary_tasks": int(features.analysis_stratum.eq("primary_ge30").sum()),
        "risk_components": list(RISK_COMPONENTS),
        "registered_family": {
            "members": [
                *[f"TxPert-STRING-GAT/seed_{seed}" for seed in SEEDS],
                *[f"TxPert-Exphormer/seed_{seed}" for seed in SEEDS],
            ],
            "architecture_total_weights": {
                "TxPert-STRING-GAT": 0.5,
                "TxPert-Exphormer": 0.5,
            },
            "member_weights": [0.125] * 8,
            "lower_bound": "registered_family_disagreement",
            "tau_grid_source": (
                "10%-90% quantiles of registered-family disagreement on the "
                "sealed 1808-task primary prediction table"
            ),
            "tau_grid": tau_grid,
        },
        "safeconf_m_formula": "(4*rank(predicted_magnitude)+rank(safeconf_e205_risk))/(5*N), within target",
        "architecture_aware_router": {
            "status": "PREREGISTERED_E217_CONFIRMATION",
            "outcome": "registered_family_rms_error",
            "magnitude": "registered_predicted_magnitude",
            "safeconf": "safeconf_e205_risk",
            "disagreement": "cross_family_disagreement",
            "formula": (
                "m + 0.50*max(s-m,0) + 0.125*max(d-m,0), "
                "where m/s/d are within-target percentile ranks"
            ),
            "development_source": (
                "E217 released E153 cross-architecture parameter landscape; "
                "fixed before E205 target-truth authorization"
            ),
        },
        "certificate_priority_router": {
            "status": "PREREGISTERED_SECONDARY",
            "outcome": "registered_family_rms_error",
            "magnitude": "registered_predicted_magnitude",
            "lower_bound": "registered_family_disagreement",
            "tau_quantile": 0.8,
            "tau": tau_q80,
            "formula": (
                "2*I(registered_family_disagreement>tau_q80) + "
                "rank(registered_predicted_magnitude)/N, within target"
            ),
            "interpretation": (
                "certified tasks first; predicted magnitude orders tasks within "
                "certified and uncertified groups"
            ),
        },
        "standardization": "within target; parameters from primary_ge30 tasks",
        "family_mean_max_abs_residual": family_residual,
        "registered_family_mean_max_abs_residual": registered_centroid_residual,
        "gat_control_max_rmse_residual": control_residual,
        "target_expression_nonzero_values_seen": 0,
        "target_truth_materialized": False,
        "target_outcomes_evaluated": False,
        "risk_table": risk_record,
        "vector_files": vector_records,
        "prediction_inputs": prediction_inputs,
    }
    atomic_json(risk_status, status)
    print(json.dumps({**status, "prediction_inputs": "16 sealed records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
