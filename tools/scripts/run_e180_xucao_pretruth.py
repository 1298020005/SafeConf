#!/usr/bin/env python3
"""Run the frozen E180 five-seed pretruth predictors and adaptive error base."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor


ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
OUT = ROOT / "docs/实验结果/E180_xucao_fresh_guide_certificate_20260723"
RELEASE = OUT / "pretruth_release"
STAGING = OUT / f".pretruth_release.staging.{os.getpid()}"
ASSET_ROOT = Path("/home/yyf/data/safeconf_e180_external/isolated/F2_pretruth")
BUILDER = ROOT / "tools/scripts/build_e180_xucao_pretruth_assets.py"
E177_RUNNER = ROOT / "tools/scripts/run_e177_sunshine_pretruth.py"
E65_SCRIPT = ROOT / "tools/scripts/run_e65_scgpt_formal_fixed_panel.py"
E107_SCRIPT = ROOT / "tools/scripts/run_e107_frangieh_context_gears.py"
GO_FILE = Path("/home/yyf/data/gears_formal_baselines_v2/frangieh_local_atlas/go.csv")
MODEL_LOCK = OUT / "MODEL_INPUT_LOCK.json"
SOURCE_LOCK = OUT / "SOURCE_LOCK.json"
STAT_LOCK = OUT / "STATISTICAL_ANALYSIS_LOCK.json"
PLAN = OUT / "PREREG_ANALYSIS_PLAN.md"
TARGETS = OUT / "manifests/E180_SELECTED_TARGETS.csv"
TASKS = OUT / "manifests/E180_GUIDE_TASK_MANIFEST.csv"
SEEDS = (3407, 3408, 3409, 3410, 3411)
N_GENES = 512
FEATURES = (
    "predicted_magnitude",
    "pair_lower_bound",
    "scgpt_magnitude",
    "gears_magnitude",
    "model_cosine",
    "ensemble_mean_abs",
    "ensemble_max_abs",
    "disagreement_mean_abs",
    "sign_agreement",
    "ensemble_abs_p90",
    "ensemble_abs_p99",
    "disagreement_abs_p90",
    "disagreement_abs_p99",
    "ensemble_gene_std",
    "disagreement_gene_std",
    "scgpt_seed_spread",
    "gears_seed_spread",
    "ensemble_seed_spread",
)
ALLOWLIST = {
    "ACCESS_ATTESTATION.json",
    "CONTROL_PROFILES.npz",
    "GENE_PANEL.csv",
    "MANIFEST.sha256",
    "PRETRUTH_TASKS.csv",
    "ROW_ACCESS_AUDIT.csv",
    "SEEN_TARGET_EFFECTS.npz",
    "TRAIN_CONTROL_COEXPRESSION_EDGES.csv",
    "TRAIN_CONTROL_PROFILE_INDEX.csv",
}


class IntegrityError(RuntimeError):
    """E180 pretruth integrity failure."""


@dataclass
class Assets:
    panel: pd.DataFrame
    tasks: pd.DataFrame
    control: np.ndarray
    seen_effects: dict[str, np.ndarray]
    coexpression: pd.DataFrame
    attestation: dict[str, Any]
    manifest_sha256: str
    input_hashes: list[dict[str, Any]]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_bytes(path, frame.to_csv(index=False, float_format="%.17g").encode())


def atomic_json(path: Path, value: object) -> None:
    atomic_bytes(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
    )


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("xb") as handle:
        np.savez_compressed(
            handle,
            **{key: np.asarray(value, np.float32) for key, value in sorted(arrays.items())},
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def git_text(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def import_script(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise IntegrityError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def require_committed(path: Path, head: str) -> dict[str, Any]:
    relative = path.relative_to(ROOT).as_posix()
    local = path.read_bytes()
    try:
        committed = subprocess.check_output(["git", "show", f"{head}:{relative}"], cwd=ROOT)
    except subprocess.CalledProcessError as exc:
        raise IntegrityError(f"uncommitted E180 input: {relative}") from exc
    if local != committed:
        raise IntegrityError(f"E180 input differs from HEAD: {relative}")
    return {"path": relative, "bytes": len(local), "sha256": hashlib.sha256(local).hexdigest()}


def verify_dual_remote(head: str) -> tuple[str, dict[str, str]]:
    branch = git_text("rev-parse", "--abbrev-ref", "HEAD")
    if branch == "HEAD":
        raise IntegrityError("E180 pretruth requires a named branch")
    result: dict[str, str] = {}
    for remote in ("origin", "github"):
        fetched = subprocess.run(
            [
                "git",
                "fetch",
                "--quiet",
                remote,
                f"refs/heads/{branch}:refs/remotes/{remote}/{branch}",
            ],
            cwd=ROOT,
            check=False,
        )
        if fetched.returncode:
            raise IntegrityError(f"cannot fetch E180 freeze from {remote}")
        remote_head = git_text("rev-parse", f"refs/remotes/{remote}/{branch}")
        if subprocess.run(
            ["git", "merge-base", "--is-ancestor", head, remote_head],
            cwd=ROOT,
            check=False,
        ).returncode:
            raise IntegrityError(f"E180 HEAD is absent from {remote}")
        result[remote] = remote_head
    return branch, result


def formal_input_audit() -> tuple[str, str, dict[str, str], list[dict[str, Any]]]:
    head = git_text("rev-parse", "HEAD")
    branch, remotes = verify_dual_remote(head)
    paths = [
        RUNNER,
        BUILDER,
        E177_RUNNER,
        E65_SCRIPT,
        E107_SCRIPT,
        MODEL_LOCK,
        SOURCE_LOCK,
        STAT_LOCK,
        PLAN,
        TARGETS,
        TASKS,
    ]
    hashes = [require_committed(path, head) for path in paths]
    lock = json.loads(MODEL_LOCK.read_text())
    for path_text, expected in lock["scgpt_checkpoint_files"].items():
        path = Path(path_text)
        if sha256_file(path) != expected:
            raise IntegrityError(f"E180 scGPT checkpoint changed: {path}")
        hashes.append({"path": str(path), "bytes": path.stat().st_size, "sha256": expected})
    vocab = Path(lock["scgpt_vocab"]["path"])
    if sha256_file(vocab) != lock["scgpt_vocab"]["sha256"]:
        raise IntegrityError("E180 scGPT vocab changed")
    go = Path(lock["gears_go_prior"]["path"])
    if sha256_file(go) != lock["gears_go_prior"]["sha256"]:
        raise IntegrityError("E180 GO prior changed")
    return head, branch, remotes, hashes


def parse_manifest(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        values[name.strip()] = digest
    if set(values) != ALLOWLIST - {"MANIFEST.sha256"}:
        raise IntegrityError("E180 F2 manifest allowlist changed")
    return values


def load_vectors(path: Path) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as archive:
        for key in archive.files:
            value = np.asarray(archive[key], np.float32)
            if value.shape != (N_GENES,) or not np.isfinite(value).all():
                raise IntegrityError(f"invalid E180 vector: {path}:{key}")
            result[str(key)] = value
    return result


def load_assets() -> Assets:
    root = ASSET_ROOT.resolve(strict=True)
    if root.name != "F2_pretruth" or root.is_symlink():
        raise IntegrityError("E180 accepts only isolated F2_pretruth")
    observed = {path.name for path in root.iterdir() if path.is_file()}
    if observed != ALLOWLIST:
        raise IntegrityError(f"E180 F2 file set changed: {sorted(observed)}")
    manifest = parse_manifest(root / "MANIFEST.sha256")
    input_hashes: list[dict[str, Any]] = []
    for name, expected in manifest.items():
        path = root / name
        if sha256_file(path) != expected:
            raise IntegrityError(f"E180 F2 hash mismatch: {name}")
        input_hashes.append({"path": str(path), "bytes": path.stat().st_size, "sha256": expected})
    panel = pd.read_csv(root / "GENE_PANEL.csv", keep_default_na=False)
    frozen_targets = pd.read_csv(TARGETS, keep_default_na=False)
    if (
        len(panel) != N_GENES
        or panel["scgpt_token"].nunique() != N_GENES
        or panel["panel_role"].eq("REGISTERED_TARGET").sum() != len(frozen_targets)
    ):
        raise IntegrityError("E180 panel schema changed")
    frozen = pd.read_csv(TASKS, keep_default_na=False)
    tasks = pd.read_csv(root / "PRETRUTH_TASKS.csv", keep_default_na=False)
    shared = list(frozen.columns)
    left = frozen[shared].sort_values("task_id").reset_index(drop=True).astype(str)
    right = tasks[shared].sort_values("task_id").reset_index(drop=True).astype(str)
    if not left.equals(right):
        raise IntegrityError("E180 F2 tasks differ from frozen guide manifest")
    controls = load_vectors(root / "CONTROL_PROFILES.npz")
    if set(controls) != {"GLOBAL"}:
        raise IntegrityError("E180 control profile changed")
    seen = load_vectors(root / "SEEN_TARGET_EFFECTS.npz")
    expected_seen = set(
        tasks.loc[
            tasks["target_split"].isin(["supervised_train", "model_validation"]),
            "task_id",
        ].astype(str)
    )
    if set(seen) != expected_seen:
        raise IntegrityError("E180 seen effects changed")
    hidden = set(
        tasks.loc[
            tasks["target_split"].isin(
                ["conformal_calibration", "prospective_evaluation"]
            ),
            "task_id",
        ].astype(str)
    )
    if set(seen) & hidden:
        raise IntegrityError("E180 hidden truth entered F2")
    coexpression = pd.read_csv(root / "TRAIN_CONTROL_COEXPRESSION_EDGES.csv")
    if coexpression.empty or not {"source", "target", "importance"}.issubset(coexpression):
        raise IntegrityError("E180 coexpression graph invalid")
    attestation = json.loads((root / "ACCESS_ATTESTATION.json").read_text())
    required = {
        "status": "PASS",
        "stage": "F2_PRETRUTH_ISOLATED_ASSET_BUILD",
        "calibration_target_x_rows_read": 0,
        "evaluation_target_x_rows_read": 0,
    }
    if any(attestation.get(key) != value for key, value in required.items()):
        raise IntegrityError("E180 F2 attestation changed")
    return Assets(
        panel,
        tasks,
        controls["GLOBAL"],
        seen,
        coexpression,
        attestation,
        sha256_file(root / "MANIFEST.sha256"),
        input_hashes,
    )


def build_graphs(assets: Assets) -> tuple[dict[str, list[Any]], list[Any], pd.DataFrame]:
    import torch
    from torch_geometric.data import Data

    target_position = {
        str(row.gene_name): int(row.panel_index)
        for row in assets.panel.itertuples(index=False)
        if str(row.panel_role) == "REGISTERED_TARGET"
    }
    supervised: dict[str, list[Any]] = {"train": [], "validation": []}
    query: list[Any] = []
    audit: list[dict[str, Any]] = []
    for row in assets.tasks.itertuples(index=False):
        task_id = str(row.task_id)
        flag = np.zeros(N_GENES, np.float32)
        flag[target_position[str(row.perturbation)]] = 1.0
        x = torch.from_numpy(np.stack([assets.control, flag], axis=1))
        query_graph = Data(
            x=x,
            pert=task_id,
            perturbation=str(row.perturbation),
            guide_id=str(row.guide_id),
            target_split=str(row.target_split),
        )
        query.append(query_graph)
        audit.append(
            {
                "task_id": task_id,
                "graph_role": "query",
                "target_split": str(row.target_split),
                "contains_y": False,
            }
        )
        if task_id in assets.seen_effects:
            graph = Data(
                x=x,
                y=torch.from_numpy(assets.control + assets.seen_effects[task_id]).unsqueeze(0),
                pert=task_id,
                perturbation=str(row.perturbation),
                guide_id=str(row.guide_id),
                target_split=str(row.target_split),
            )
            split = {
                "supervised_train": "train",
                "model_validation": "validation",
            }.get(str(row.target_split))
            if split is None:
                raise IntegrityError("E180 hidden task has supervised y")
            supervised[split].append(graph)
            audit.append(
                {
                    "task_id": task_id,
                    "graph_role": f"supervised_{split}",
                    "target_split": str(row.target_split),
                    "contains_y": True,
                }
            )
    expected_train = int(assets.tasks["target_split"].eq("supervised_train").sum())
    expected_validation = int(assets.tasks["target_split"].eq("model_validation").sum())
    if len(supervised["train"]) != expected_train or len(supervised["validation"]) != expected_validation:
        raise IntegrityError("E180 supervised graph counts changed")
    return supervised, query, pd.DataFrame(audit)


def vector_features(scgpt: np.ndarray, gears: np.ndarray) -> pd.DataFrame:
    ensemble = (scgpt + gears) / 2.0
    difference = scgpt - gears
    rms = lambda value: np.sqrt(np.mean(np.square(value, dtype=np.float64), axis=1))
    denominator = np.linalg.norm(scgpt, axis=1) * np.linalg.norm(gears, axis=1)
    return pd.DataFrame(
        {
            "predicted_magnitude": rms(ensemble),
            "pair_lower_bound": rms(difference) / 2.0,
            "scgpt_magnitude": rms(scgpt),
            "gears_magnitude": rms(gears),
            "model_cosine": np.sum(scgpt * gears, axis=1) / np.maximum(denominator, 1e-12),
            "ensemble_mean_abs": np.mean(np.abs(ensemble), axis=1),
            "ensemble_max_abs": np.max(np.abs(ensemble), axis=1),
            "disagreement_mean_abs": np.mean(np.abs(difference), axis=1),
            "sign_agreement": np.mean(np.sign(scgpt) == np.sign(gears), axis=1),
            "ensemble_abs_p90": np.quantile(np.abs(ensemble), 0.90, axis=1),
            "ensemble_abs_p99": np.quantile(np.abs(ensemble), 0.99, axis=1),
            "disagreement_abs_p90": np.quantile(np.abs(difference), 0.90, axis=1),
            "disagreement_abs_p99": np.quantile(np.abs(difference), 0.99, axis=1),
            "ensemble_gene_std": np.std(ensemble, axis=1),
            "disagreement_gene_std": np.std(difference, axis=1),
        }
    )


def assemble(
    assets: Assets, predictions: dict[str, dict[str, np.ndarray]]
) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    task_order = assets.tasks["task_id"].astype(str).tolist()
    expected = {
        *(f"scGPT_seed{seed}" for seed in SEEDS),
        *(f"GEARS_seed{seed}" for seed in SEEDS),
    }
    if set(predictions) != expected:
        raise IntegrityError("E180 prediction family changed")
    arrays = {
        name: np.stack([mapping[task] for task in task_order]).astype(np.float32)
        for name, mapping in predictions.items()
    }
    arrays["scGPT_seed_mean"] = np.mean(
        np.stack([arrays[f"scGPT_seed{seed}"] for seed in SEEDS]), axis=0
    ).astype(np.float32)
    arrays["GEARS_seed_mean"] = np.mean(
        np.stack([arrays[f"GEARS_seed{seed}"] for seed in SEEDS]), axis=0
    ).astype(np.float32)
    arrays["ensemble_seed_family_mean"] = (
        (arrays["scGPT_seed_mean"] + arrays["GEARS_seed_mean"]) / 2.0
    ).astype(np.float32)
    features = vector_features(arrays["scGPT_seed_mean"], arrays["GEARS_seed_mean"])
    sc_seeds = np.stack([arrays[f"scGPT_seed{seed}"] for seed in SEEDS])
    ge_seeds = np.stack([arrays[f"GEARS_seed{seed}"] for seed in SEEDS])
    features["scgpt_seed_spread"] = np.sqrt(np.mean(np.var(sc_seeds, axis=0), axis=1))
    features["gears_seed_spread"] = np.sqrt(np.mean(np.var(ge_seeds, axis=0), axis=1))
    features["ensemble_seed_spread"] = np.sqrt(
        np.mean(np.var((sc_seeds + ge_seeds) / 2.0, axis=0), axis=1)
    )
    scores = pd.concat([assets.tasks.reset_index(drop=True), features], axis=1)
    scores["constant_base"] = 0.0
    scores["magnitude_base"] = scores["predicted_magnitude"]
    scores["magnitude_plus_lower_base"] = np.maximum(
        scores["predicted_magnitude"], scores["pair_lower_bound"]
    )
    scores["validation_pair_mean_rmse"] = np.nan
    validation_rows: list[dict[str, Any]] = []
    for index, row in scores.iterrows():
        task_id = str(row["task_id"])
        if str(row["target_split"]) != "model_validation":
            continue
        truth = assets.seen_effects[task_id]
        sc_error = float(np.sqrt(np.mean((arrays["scGPT_seed_mean"][index] - truth) ** 2)))
        ge_error = float(np.sqrt(np.mean((arrays["GEARS_seed_mean"][index] - truth) ** 2)))
        pair_mean = (sc_error + ge_error) / 2.0
        scores.loc[index, "validation_pair_mean_rmse"] = pair_mean
        validation_rows.append(
            {
                "task_id": task_id,
                "perturbation": row["perturbation"],
                "guide_id": row["guide_id"],
                "scgpt_rmse": sc_error,
                "gears_rmse": ge_error,
                "pair_mean_rmse": pair_mean,
                "pair_lower_bound": row["pair_lower_bound"],
            }
        )
    validation = scores["target_split"].eq("model_validation")
    if scores.loc[validation, "validation_pair_mean_rmse"].isna().any():
        raise IntegrityError("E180 validation errors incomplete")
    model = ExtraTreesRegressor(
        n_estimators=200,
        min_samples_leaf=10,
        max_features=0.70,
        random_state=3407,
        n_jobs=-1,
    )
    model.fit(
        scores.loc[validation, FEATURES].to_numpy(float),
        scores.loc[validation, "validation_pair_mean_rmse"].to_numpy(float),
    )
    scores["extra_trees_vector_base"] = np.maximum(
        scores["pair_lower_bound"].to_numpy(float),
        model.predict(scores.loc[:, FEATURES].to_numpy(float)),
    )
    scores["calibration_or_evaluation_truth_present"] = False
    scores["adaptive_base_training_truth_role"] = np.where(
        validation, "MODEL_VALIDATION_ONLY", ""
    )
    importance = pd.DataFrame(
        {"feature": FEATURES, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    return scores, arrays, pd.DataFrame(validation_rows), importance


def run_gates(
    assets: Assets,
    scores: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    graph_audit: pd.DataFrame,
    e177: Any,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    hidden = scores["target_split"].isin(
        ["conformal_calibration", "prospective_evaluation"]
    )
    checks = {
        "G1_hidden_query_graphs_have_no_y": not (
            graph_audit["graph_role"].eq("query")
            & graph_audit["contains_y"].astype(bool)
            & graph_audit["target_split"].isin(
                ["conformal_calibration", "prospective_evaluation"]
            )
        ).any(),
        "G2_hidden_truth_absent_from_scoring_interface": not scores.loc[
            hidden, "calibration_or_evaluation_truth_present"
        ].astype(bool).any(),
        "G3_adaptive_base_finite_and_variable": bool(
            np.isfinite(scores["extra_trees_vector_base"]).all()
            and scores.loc[hidden, "extra_trees_vector_base"].nunique() > 2
        ),
        "G4_pair_lower_finite": bool(
            np.isfinite(scores["pair_lower_bound"]).all()
            and (scores["pair_lower_bound"] >= 0).all()
        ),
        "G5_predictor_vectors_variable": bool(
            e177.predictor_gate(arrays["scGPT_seed_mean"][hidden.to_numpy()])["passed"]
            and e177.predictor_gate(arrays["GEARS_seed_mean"][hidden.to_numpy()])["passed"]
        ),
        "G6_five_seed_predictions_complete": all(
            arrays[f"{model}_seed{seed}"].shape == (len(scores), N_GENES)
            for model in ("scGPT", "GEARS")
            for seed in SEEDS
        ),
        "G7_f2_hidden_x_rows_zero": bool(
            assets.attestation["calibration_target_x_rows_read"] == 0
            and assets.attestation["evaluation_target_x_rows_read"] == 0
        ),
    }
    for gate, passed in checks.items():
        rows.append({"gate": gate, "passed": passed})
    return pd.DataFrame(rows)


def runtime_environment(device: Any) -> dict[str, Any]:
    import torch
    import torch_geometric

    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "cuda_runtime": torch.version.cuda,
        "selected_device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
    }


def write_release(
    assets: Assets,
    scores: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    validation_errors: pd.DataFrame,
    importance: pd.DataFrame,
    graph_audit: pd.DataFrame,
    histories: pd.DataFrame,
    model_audit: pd.DataFrame,
    gates: pd.DataFrame,
    audit: tuple[str, str, dict[str, str], list[dict[str, Any]]],
    environment: dict[str, Any],
    wall_seconds: float,
) -> Path:
    head, branch, remotes, hashes = audit
    if RELEASE.exists() or STAGING.exists():
        raise IntegrityError("E180 pretruth release is append-only")
    try:
        for sub in ("tables", "arrays", "reports"):
            (STAGING / sub).mkdir(parents=True, exist_ok=False)
        atomic_csv(STAGING / "tables/PRETRUTH_SCORING_INTERFACE.csv", scores)
        atomic_npz(STAGING / "arrays/PRETRUTH_PREDICTIONS.npz", arrays)
        atomic_csv(STAGING / "tables/VALIDATION_TASK_ERRORS.csv", validation_errors)
        atomic_csv(STAGING / "tables/ADAPTIVE_BASE_FEATURE_IMPORTANCE.csv", importance)
        atomic_csv(STAGING / "tables/QUERY_GRAPH_AUDIT.csv", graph_audit)
        atomic_csv(STAGING / "tables/TRAINING_HISTORY.csv", histories)
        atomic_csv(STAGING / "tables/MODEL_AUDIT.csv", model_audit)
        atomic_csv(STAGING / "tables/PRETRUTH_GATES.csv", gates)
        atomic_csv(
            STAGING / "tables/INPUT_HASHES.csv",
            pd.DataFrame(hashes + assets.input_hashes),
        )
        atomic_json(STAGING / "RUNTIME_ENVIRONMENT.json", environment)
        snapshot = {
            "schema": "safeconf_e180_pretruth_snapshot_v1",
            "experiment": "E180_xucao_fresh_guide_certificate",
            "status": "PASS" if gates["passed"].astype(bool).all() else "FAIL",
            "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
            "git_head": head,
            "git_branch": branch,
            "code_freeze_remote_heads": remotes,
            "runner_sha256": sha256_file(RUNNER),
            "builder_sha256": sha256_file(BUILDER),
            "f2_manifest_sha256": assets.manifest_sha256,
            "n_tasks": len(scores),
            "n_train_tasks": int(scores["target_split"].eq("supervised_train").sum()),
            "n_validation_tasks": int(scores["target_split"].eq("model_validation").sum()),
            "n_calibration_query_tasks": int(
                scores["target_split"].eq("conformal_calibration").sum()
            ),
            "n_evaluation_query_tasks": int(
                scores["target_split"].eq("prospective_evaluation").sum()
            ),
            "calibration_target_x_rows_read": 0,
            "evaluation_target_x_rows_read": 0,
            "adaptive_base_fitted_from": "model_validation_targets_only",
            "cell_cycle_phase_used_as_primary_context": False,
            "wall_seconds": wall_seconds,
        }
        atomic_json(STAGING / "PRETRUTH_GATE_SNAPSHOT.json", snapshot)
        report = (
            "# E180 预真值模型报告\n\n"
            f"状态：**{snapshot['status']}**。\n\n"
            f"监督训练任务 {snapshot['n_train_tasks']} 条，模型验证任务 "
            f"{snapshot['n_validation_tasks']} 条，calibration 查询 "
            f"{snapshot['n_calibration_query_tasks']} 条，evaluation 查询 "
            f"{snapshot['n_evaluation_query_tasks']} 条。\n\n"
            "scGPT、GEARS 五随机种子预测和 ExtraTrees 误差基线已经冻结。"
            "calibration/evaluation 靶点表达值仍未读取；下一步先提交并双远端推送本目录，"
            "再单独打开 calibration 真值。\n"
        )
        atomic_bytes(STAGING / "reports/E180_PRETRUTH_REPORT.md", report.encode())
        os.replace(STAGING, RELEASE)
        return RELEASE / "PRETRUTH_GATE_SNAPSHOT.json"
    except Exception:
        shutil.rmtree(STAGING, ignore_errors=True)
        raise


def run_formal(device_name: str) -> dict[str, Any]:
    started = time.time()
    audit = formal_input_audit()
    assets = load_assets()
    supervised, query, graph_audit = build_graphs(assets)
    e177 = import_script("e177_helpers_for_e180", E177_RUNNER)
    e177.N_GENES = N_GENES
    e177.E65_SCRIPT = E65_SCRIPT
    e177.E107_SCRIPT = E107_SCRIPT
    e177.GO_FILE = GO_FILE
    import torch

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise IntegrityError("E180 registered CUDA device unavailable")
    genes = assets.panel["scgpt_token"].astype(str).tolist()
    predictions: dict[str, dict[str, np.ndarray]] = {}
    histories: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for seed in SEEDS:
        scgpt, history, model_info = e177.train_scgpt(
            seed, supervised, query, genes, device
        )
        predictions[f"scGPT_seed{seed}"] = scgpt
        histories.append(history)
        audits.append({"seed": seed, "model": "scGPT", **model_info})
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gears, history, model_info = e177.train_gears(
            seed, supervised, query, genes, assets.coexpression, device
        )
        predictions[f"GEARS_seed{seed}"] = gears
        histories.append(history)
        audits.append({"seed": seed, "model": "GEARS", **model_info})
        if device.type == "cuda":
            torch.cuda.empty_cache()
    scores, arrays, validation_errors, importance = assemble(assets, predictions)
    gates = run_gates(assets, scores, arrays, graph_audit, e177)
    snapshot = write_release(
        assets,
        scores,
        arrays,
        validation_errors,
        importance,
        graph_audit,
        pd.concat(histories, ignore_index=True),
        pd.DataFrame(audits),
        gates,
        audit,
        runtime_environment(device),
        time.time() - started,
    )
    return {
        "status": "PASS" if gates["passed"].astype(bool).all() else "FAIL",
        "snapshot": str(snapshot.relative_to(ROOT)),
        "snapshot_sha256": sha256_file(snapshot),
        "wall_seconds": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--asset-audit-only", action="store_true")
    args = parser.parse_args()
    if args.asset_audit_only:
        assets = load_assets()
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "n_tasks": len(assets.tasks),
                    "n_seen_effects": len(assets.seen_effects),
                    "f2_manifest_sha256": assets.manifest_sha256,
                },
                indent=2,
            )
        )
        return
    result = run_formal(args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
