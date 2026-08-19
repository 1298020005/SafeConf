#!/usr/bin/env python3
"""E197: post-truth Systema/scPertEval centroid-only audit of E190/E192.

The default ``prepare`` mode hashes and validates metadata without loading the
prediction or truth arrays.  The formal analysis requires the explicit
``--allow-posttruth-evaluation`` acknowledgement because both source
experiments have already unsealed their target truth.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
OUT = (
    ROOT
    / "docs/实验结果/E197_systema_scperteval_centroid_audit_20260730"
)
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
FREEZE = OUT / "ANALYSIS_FREEZE.md"
SCRIPT = Path(__file__).resolve()
SCPERTEVAL_REPO = Path("/home/yyf/archive/external/scPertEval")
SCPERTEVAL_COMMIT = "8709eb07a0e7d4ecf1c60c977f2018690a749975"
SYSTEMA_REPO = Path("/home/yyf/archive/external/Systema")
SYSTEMA_COMMIT = "aaf5b5353993b48b78543f2f93b3e18ca65df515"
N_GENES = 512
MODEL_KEYS = tuple(
    f"{architecture}_seed{seed}"
    for seed in (3407, 3408, 3409)
    for architecture in ("scGPT", "GEARS")
)
ALLOWED_SCPERTEVAL_PROTOCOLS = (
    "pearson",
    "pearson_ctrl",
    "pearson_pert",
    "mse",
    "rank",
    "transpose_rank",
)
FORBIDDEN_PROTOCOL_FAMILIES = (
    "mmd",
    "energy",
    "sinkhorn",
    "de_",
    "wmse",
    "top_k",
    "degs",
)
N_BOOTSTRAP = 5000
EXPECTED_PREDICTORS = MODEL_KEYS + (
    "family_centroid",
    "matching_source_train_mean",
    "matching_source_all_folds_mean",
    "target_control_plus_source_mean_effect",
    "source_absolute_noncontrol_mean",
    "zero_effect",
)
N_PREDICTORS = len(EXPECTED_PREDICTORS)
OUTPUT_HASH_INDEX = TABLES / "E197_OUTPUT_HASHES.csv"


SETTINGS: dict[str, dict[str, Any]] = {
    "E190_K562": {
        "eid": "E190",
        "root": ROOT
        / "docs/实验结果/E190_adamson_to_replogle_direct_transfer_20260729",
        "assets": Path(
            "/home/yyf/data/safeconf_e190_adamson_replogle/model_assets"
        ),
        "n_tasks": 692,
        "n_genes": 47,
    },
    "E192_RPE1": {
        "eid": "E192",
        "root": ROOT
        / "docs/实验结果/E192_adamson_to_replogle_rpe1_locked_transfer_20260729",
        "assets": Path(
            "/home/yyf/data/safeconf_e192_adamson_rpe1/model_assets"
        ),
        "n_tasks": 175,
        "n_genes": 21,
    },
}


class ContractFailure(RuntimeError):
    """Fail-closed E197 contract error."""


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logical_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        data_root = Path("/home/yyf/data").resolve()
        try:
            return "DATA/" + resolved.relative_to(data_root).as_posix()
        except ValueError:
            try:
                return (
                    "EXTERNAL/"
                    + resolved.relative_to(Path("/home/yyf/archive/external")).as_posix()
                )
            except ValueError:
                return resolved.name


def add_hash(rows: list[dict[str, Any]], role: str, path: Path) -> None:
    if not path.is_file():
        raise ContractFailure(f"missing input: {path}")
    rows.append(
        {
            "role": role,
            "path": logical_path(path),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
    )


def write_text_atomic(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload)
    temporary.replace(path)


def write_status(status: str, **fields: Any) -> None:
    payload = {
        "experiment": "E197_systema_scperteval_centroid_audit",
        "generated_at": now(),
        "status": status,
        "analysis_class": "POSTTRUTH_EXPLORATORY",
        **fields,
    }
    write_text_atomic(
        OUT / "E197_STATUS.json",
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )


def output_hashes(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(paths, key=lambda item: logical_path(item)):
        add_hash(rows, "E197:output", path)
    frame = pd.DataFrame(rows)
    temporary = OUTPUT_HASH_INDEX.with_name(f".{OUTPUT_HASH_INDEX.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(OUTPUT_HASH_INDEX)
    return frame


def verify_release_locks(
    rows: list[dict[str, Any]], role: str, release_root: Path, lock_path: Path
) -> None:
    locks = pd.read_csv(lock_path, keep_default_na=False)
    if locks.columns.tolist() != ["path", "bytes", "sha256"]:
        raise ContractFailure(f"unexpected lock schema: {lock_path}")
    for item in locks.itertuples(index=False):
        relative = Path(str(item.path))
        if relative.is_absolute() or ".." in relative.parts:
            raise ContractFailure(f"unsafe locked path: {relative}")
        path = (release_root / relative).resolve()
        try:
            path.relative_to(release_root.resolve())
        except ValueError as exc:
            raise ContractFailure(f"locked path escapes root: {path}") from exc
        observed = sha256_file(path)
        if observed != str(item.sha256) or path.stat().st_size != int(item.bytes):
            raise ContractFailure(f"source release lock mismatch: {path}")
        rows.append(
            {
                "role": f"{role}:{relative.as_posix()}",
                "path": logical_path(path),
                "bytes": int(path.stat().st_size),
                "sha256": observed,
            }
        )


def gate(
    rows: list[dict[str, Any]],
    setting: str,
    check: str,
    observed: Any,
    expected: Any,
    passed: bool,
    detail: str = "",
) -> None:
    rows.append(
        {
            "setting": setting,
            "check": check,
            "observed": observed,
            "expected": expected,
            "passed": bool(passed),
            "detail": detail,
        }
    )
    if not passed:
        raise ContractFailure(f"{setting} gate failed: {check}")


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).strip()


def metadata_audit(
    setting: str, spec: dict[str, Any], hashes: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str], dict[str, str]]:
    """Audit metadata and locks without opening prediction/truth array contents."""
    eid = str(spec["eid"])
    root = Path(spec["root"])
    assets = Path(spec["assets"])
    final = root / "final_evaluation"
    pred = root / "pretruth_release"
    truth = root / "evaluation_truth"
    gates: list[dict[str, Any]] = []

    core_files = [
        root / f"{eid}_SELECTED_GENES.csv",
        final / "tables" / f"{eid}_TASK_METRICS.csv",
        pred / "tables/QUERY_ORDER.csv",
        truth / "tables/TARGET_TRUTH_INDEX.csv",
        pred / "RELEASE_LOCKS.csv",
        truth / "TRUTH_LOCKS.csv",
        pred / "PRETRUTH_STATUS.json",
        truth / "TARGET_TRUTH_BUILD_STATUS.json",
    ]
    asset_files = [
        assets / "ASSET_MANIFEST.json",
        assets / "GENE_PANEL.csv",
        assets / "QUERY_TASKS.csv",
        assets / "TRAIN_TASKS.csv",
        assets / "TRAIN_EFFECTS.npz",
        assets / "SOURCE_CONTROL_PROFILE.npz",
        assets / "SOURCE_GENE_EFFECTS.npz",
        assets / "TARGET_CONTROL_PROFILES.npz",
    ]
    for path in core_files + asset_files:
        add_hash(hashes, f"{setting}:{path.name}", path)
    verify_release_locks(
        hashes, f"{setting}:prediction_release", pred, pred / "RELEASE_LOCKS.csv"
    )
    verify_release_locks(
        hashes, f"{setting}:truth_release", truth, truth / "TRUTH_LOCKS.csv"
    )

    selected = pd.read_csv(root / f"{eid}_SELECTED_GENES.csv", keep_default_na=False)
    task_metrics = pd.read_csv(
        final / "tables" / f"{eid}_TASK_METRICS.csv", keep_default_na=False
    )
    order = pd.read_csv(pred / "tables/QUERY_ORDER.csv", keep_default_na=False)
    truth_index = pd.read_csv(
        truth / "tables/TARGET_TRUTH_INDEX.csv", keep_default_na=False
    )
    query = pd.read_csv(assets / "QUERY_TASKS.csv", keep_default_na=False)
    panel = pd.read_csv(assets / "GENE_PANEL.csv", keep_default_na=False)
    train = pd.read_csv(assets / "TRAIN_TASKS.csv", keep_default_na=False)
    pred_status = json.loads((pred / "PRETRUTH_STATUS.json").read_text())
    truth_status = json.loads((truth / "TARGET_TRUTH_BUILD_STATUS.json").read_text())

    expected_tasks = int(spec["n_tasks"])
    expected_genes = int(spec["n_genes"])
    task_ids = order.task_id.astype(str).tolist()
    gate(gates, setting, "task_count", len(order), expected_tasks, len(order) == expected_tasks)
    gate(
        gates,
        setting,
        "selected_gene_count",
        selected.gene.astype(str).nunique(),
        expected_genes,
        selected.gene.astype(str).nunique() == expected_genes,
    )
    gate(
        gates,
        setting,
        "panel_gene_count",
        len(panel),
        N_GENES,
        len(panel) == N_GENES and panel.gene_name.astype(str).is_unique,
    )
    gate(
        gates,
        setting,
        "query_order_contiguous",
        order.query_index.astype(int).tolist()[:3]
        + order.query_index.astype(int).tolist()[-3:],
        f"0..{expected_tasks - 1}",
        order.query_index.astype(int).tolist() == list(range(expected_tasks)),
    )
    gate(
        gates,
        setting,
        "task_order_matches_truth_index",
        len(truth_index),
        expected_tasks,
        truth_index.task_id.astype(str).tolist() == task_ids,
    )
    gate(
        gates,
        setting,
        "task_order_matches_asset_query",
        len(query),
        expected_tasks,
        query.task_id.astype(str).tolist() == task_ids,
    )
    gate(
        gates,
        setting,
        "task_order_matches_saved_metrics",
        len(task_metrics),
        expected_tasks,
        task_metrics.task_id.astype(str).tolist() == task_ids,
    )
    selected_set = set(selected.gene.astype(str))
    task_gene_set = set(query.gene.astype(str))
    gate(
        gates,
        setting,
        "task_genes_within_selected_genes",
        len(task_gene_set),
        len(selected_set),
        task_gene_set == selected_set,
    )
    gate(
        gates,
        setting,
        "task_ids_unique",
        order.task_id.astype(str).nunique(),
        expected_tasks,
        order.task_id.astype(str).nunique() == expected_tasks,
    )
    gate(
        gates,
        setting,
        "batch_gene_tasks_unique",
        int(query.drop_duplicates(["batch", "gene"]).shape[0]),
        expected_tasks,
        not query.duplicated(["batch", "gene"]).any(),
        "context-stratified centroid competitors must be distinct perturbation genes",
    )
    gate(
        gates,
        setting,
        "prediction_release_status",
        pred_status.get("status"),
        "PASS",
        pred_status.get("status") == "PASS"
        and int(pred_status.get("target_perturbation_x_rows_read", -1)) == 0,
        "prediction release must precede target-truth read",
    )
    gate(
        gates,
        setting,
        "truth_release_status",
        truth_status.get("status"),
        "PASS",
        truth_status.get("status") == "PASS"
        and int(truth_status.get("n_target_tasks", -1)) == expected_tasks,
    )
    gate(
        gates,
        setting,
        "train_only_systema_inputs",
        sorted(train.split.astype(str).unique().tolist()),
        ["train"],
        set(train.split.astype(str)) == {"train"}
        and train.task_id.astype(str).is_unique,
    )
    required_risk = {
        "diversity_lower_bound",
        "diameter_half_lower_bound",
        "predicted_magnitude",
        "source_effect_magnitude",
    }
    gate(
        gates,
        setting,
        "frozen_risk_columns_present",
        len(required_risk & set(task_metrics.columns)),
        len(required_risk),
        required_risk <= set(task_metrics.columns),
    )
    source_hashes = {
        name: sha256_file(assets / name)
        for name in (
            "GENE_PANEL.csv",
            "TRAIN_TASKS.csv",
            "TRAIN_EFFECTS.npz",
            "SOURCE_CONTROL_PROFILE.npz",
            "SOURCE_GENE_EFFECTS.npz",
        )
    }
    return gates, panel.gene_name.astype(str).tolist(), source_hashes


def collect_prepare() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not FREEZE.is_file():
        raise ContractFailure("E197 analysis freeze is missing")
    hashes: list[dict[str, Any]] = []
    add_hash(hashes, "E197:analysis_freeze", FREEZE)
    add_hash(hashes, "E197:runner", SCRIPT)
    all_gates: list[dict[str, Any]] = []
    panels: dict[str, list[str]] = {}
    source_hashes: dict[str, dict[str, str]] = {}
    for setting, spec in SETTINGS.items():
        gates, panel, source = metadata_audit(setting, spec, hashes)
        all_gates.extend(gates)
        panels[setting] = panel
        source_hashes[setting] = source

    e190_panel = panels["E190_K562"]
    e192_panel = panels["E192_RPE1"]
    panel_overlap = len(set(e190_panel) & set(e192_panel))
    gate(
        all_gates,
        "cross_setting",
        "setting_specific_512_gene_axes_recorded",
        f"E190={len(e190_panel)};E192={len(e192_panel)};overlap={panel_overlap}",
        "two unique 512-gene axes; no cross-axis vector comparison",
        len(e190_panel) == N_GENES
        and len(e192_panel) == N_GENES
        and len(set(e190_panel)) == N_GENES
        and len(set(e192_panel)) == N_GENES,
    )
    gate(
        all_gates,
        "cross_setting",
        "source_assets_kept_setting_specific",
        int(source_hashes["E190_K562"] == source_hashes["E192_RPE1"]),
        0,
        source_hashes["E190_K562"] != source_hashes["E192_RPE1"],
        "the two target settings froze different 512-gene panels",
    )
    observed_commit = git_output(SCPERTEVAL_REPO, "rev-parse", "HEAD")
    dirty = git_output(SCPERTEVAL_REPO, "status", "--short")
    gate(
        all_gates,
        "software",
        "scperteval_official_commit",
        observed_commit,
        SCPERTEVAL_COMMIT,
        observed_commit == SCPERTEVAL_COMMIT and dirty == "",
        "official checkout must be clean",
    )
    systema_commit = git_output(SYSTEMA_REPO, "rev-parse", "HEAD")
    systema_dirty = git_output(SYSTEMA_REPO, "status", "--short")
    gate(
        all_gates,
        "software",
        "systema_official_commit",
        systema_commit,
        SYSTEMA_COMMIT,
        systema_commit == SYSTEMA_COMMIT and systema_dirty == "",
        "official Systema checkout must be clean",
    )
    add_hash(
        hashes,
        "scPertEval:protocol_table",
        SCPERTEVAL_REPO / "src/scperteval/protocols/table.py",
    )
    add_hash(
        hashes,
        "scPertEval:metric_implementation",
        SCPERTEVAL_REPO / "src/scperteval/protocols/metrics.py",
    )
    add_hash(
        hashes,
        "scPertEval:score_runner",
        SCPERTEVAL_REPO / "src/scperteval/runner.py",
    )
    add_hash(
        hashes,
        "Systema:reference_implementation",
        SYSTEMA_REPO / "evaluation/eval_utils.py",
    )
    add_hash(
        hashes,
        "Systema:pearson_delta_implementation",
        SYSTEMA_REPO / "evaluation/pearson_delta_reference_metrics.py",
    )
    add_hash(
        hashes,
        "Systema:centroid_accuracy_implementation",
        SYSTEMA_REPO / "evaluation/centroid_accuracy.py",
    )
    gate(
        all_gates,
        "protocol",
        "centroid_only_allowlist",
        ",".join(ALLOWED_SCPERTEVAL_PROTOCOLS),
        ",".join(ALLOWED_SCPERTEVAL_PROTOCOLS),
        not any(
            forbidden in protocol
            for protocol in ALLOWED_SCPERTEVAL_PROTOCOLS
            for forbidden in FORBIDDEN_PROTOCOL_FAMILIES
        ),
    )
    hash_frame = pd.DataFrame(hashes).drop_duplicates(
        subset=["role", "path", "sha256"]
    )
    hash_frame = hash_frame.sort_values(["role", "path"]).reset_index(drop=True)
    gate_frame = pd.DataFrame(all_gates)
    return hash_frame, gate_frame


def prepare() -> None:
    status_path = OUT / "E197_STATUS.json"
    if status_path.is_file():
        previous = json.loads(status_path.read_text())
        if str(previous.get("status", "")).startswith("COMPLETE"):
            raise ContractFailure("E197 is already COMPLETE; prepare will not overwrite it")
    TABLES.mkdir(parents=True, exist_ok=True)
    hashes, gates = collect_prepare()
    hashes.to_csv(TABLES / "E197_INPUT_HASHES.csv", index=False)
    gates.to_csv(TABLES / "E197_PREPARE_GATES.csv", index=False)
    status = {
        "prediction_or_truth_array_payloads_loaded": False,
        "prediction_or_truth_release_files_hashed_as_opaque_bytes": True,
        "metadata_gates_passed": int(gates.passed.astype(bool).sum()),
        "metadata_gates_total": int(len(gates)),
        "formal_run_requires": "--allow-posttruth-evaluation",
        "scperteval_commit": SCPERTEVAL_COMMIT,
        "systema_commit": SYSTEMA_COMMIT,
        "allowed_scperteval_protocols": list(ALLOWED_SCPERTEVAL_PROTOCOLS),
        "population_or_de_protocols_run": False,
    }
    write_status("PREPARED_NOT_RUN", **status)
    print(json.dumps({"status": "PREPARED_NOT_RUN", **status}, ensure_ascii=False, indent=2))


def safe_pearson(left: np.ndarray, right: np.ndarray) -> tuple[float, str]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        return math.nan, "shape_mismatch"
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return math.nan, "nonfinite"
    if float(np.std(left)) <= 1e-12:
        return math.nan, "prediction_constant"
    if float(np.std(right)) <= 1e-12:
        return math.nan, "truth_constant"
    return float(np.corrcoef(left, right)[0, 1]), ""


def safe_cosine(left: np.ndarray, right: np.ndarray) -> tuple[float, str]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        return math.nan, "shape_mismatch"
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return math.nan, "nonfinite"
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return math.nan, "zero_norm"
    return float(np.dot(left, right) / denominator), ""


def rmse(left: np.ndarray, right: np.ndarray) -> float:
    delta = np.asarray(left, dtype=np.float64) - np.asarray(
        right, dtype=np.float64
    )
    return (
        float(np.sqrt(np.mean(delta * delta)))
        if delta.shape and np.isfinite(delta).all()
        else math.nan
    )


def stable_top20(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (N_GENES,) or not np.isfinite(values).all():
        raise ContractFailure("top20 selector received malformed vector")
    return np.argsort(-np.abs(values), kind="stable")[:20]


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    keep = np.isfinite(left) & np.isfinite(right)
    if (
        keep.sum() < 4
        or np.unique(left[keep]).size < 2
        or np.unique(right[keep]).size < 2
    ):
        return math.nan
    return float(
        np.corrcoef(
            rankdata(left[keep], method="average"),
            rankdata(right[keep], method="average"),
        )[0, 1]
    )


def load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            str(key): np.asarray(archive[key], dtype=np.float64)
            for key in archive.files
        }


def systema_reference(
    assets: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    pd.DataFrame,
]:
    train = pd.read_csv(assets / "TRAIN_TASKS.csv", keep_default_na=False)
    effects = load_npz_dict(assets / "TRAIN_EFFECTS.npz")
    control_map = load_npz_dict(assets / "SOURCE_CONTROL_PROFILE.npz")
    if set(control_map) != {"SOURCE::K562"}:
        raise ContractFailure("unexpected source control keys")
    control = control_map["SOURCE::K562"]
    guide_rows: list[dict[str, Any]] = []
    guide_effects: list[np.ndarray] = []
    guide_effects_by_gene: dict[str, list[np.ndarray]] = {}
    for (gene, guide), block in train.groupby(
        ["gene", "guide"], sort=True, observed=True
    ):
        vectors = np.stack(
            [effects[str(task_id)] for task_id in block.task_id.astype(str)]
        )
        weights = block.n_cells.to_numpy(dtype=np.float64)
        if len(block) != 4 or np.any(weights <= 0):
            raise ContractFailure(f"malformed train guide pseudoreplicates: {guide}")
        effect = np.average(vectors, axis=0, weights=weights)
        guide_effects.append(effect)
        guide_effects_by_gene.setdefault(str(gene), []).append(effect)
        guide_rows.append(
            {
                "gene": str(gene),
                "guide": str(guide),
                "n_train_pseudoreplicates": int(len(block)),
                "n_train_cells": int(weights.sum()),
                "guide_effect_rms": rmse(effect, np.zeros_like(effect)),
            }
        )
    mean_effect = np.mean(np.stack(guide_effects), axis=0)
    reference = control + mean_effect
    if (
        reference.shape != (N_GENES,)
        or mean_effect.shape != (N_GENES,)
        or not np.isfinite(reference).all()
        or not np.isfinite(mean_effect).all()
    ):
        raise ContractFailure("Systema reference reconstruction failed")
    train_gene_effects = {
        gene: np.mean(np.stack(vectors), axis=0)
        for gene, vectors in guide_effects_by_gene.items()
    }
    if any(
        vector.shape != (N_GENES,) or not np.isfinite(vector).all()
        for vector in train_gene_effects.values()
    ):
        raise ContractFailure("train-only matching effects are malformed")
    return (
        reference,
        control,
        mean_effect,
        train_gene_effects,
        pd.DataFrame(guide_rows),
    )


def load_setting(setting: str, spec: dict[str, Any]) -> dict[str, Any]:
    eid = str(spec["eid"])
    root = Path(spec["root"])
    assets = Path(spec["assets"])
    order = pd.read_csv(
        root / "pretruth_release/tables/QUERY_ORDER.csv", keep_default_na=False
    )
    query = pd.read_csv(assets / "QUERY_TASKS.csv", keep_default_na=False)
    saved = pd.read_csv(
        root / f"final_evaluation/tables/{eid}_TASK_METRICS.csv",
        keep_default_na=False,
    )
    panel = pd.read_csv(assets / "GENE_PANEL.csv", keep_default_na=False)
    task_ids = order.task_id.astype(str).tolist()
    if (
        query.task_id.astype(str).tolist() != task_ids
        or saved.task_id.astype(str).tolist() != task_ids
        or len(task_ids) != int(spec["n_tasks"])
        or len(panel) != N_GENES
    ):
        raise ContractFailure(f"{setting}: formal task/gene order gate failed")

    with np.load(
        root / "pretruth_release/arrays/PRETRUTH_PREDICTIONS.npz",
        allow_pickle=False,
    ) as archive:
        if set(archive.files) != set(MODEL_KEYS):
            raise ContractFailure(f"{setting}: model key mismatch")
        predictions = np.stack(
            [
                np.asarray(archive[key], dtype=np.float64)
                for key in MODEL_KEYS
            ]
        )
    truth_map = load_npz_dict(
        root / "evaluation_truth/arrays/TARGET_TRUE_EFFECTS.npz"
    )
    if set(truth_map) != set(task_ids):
        raise ContractFailure(f"{setting}: truth task key mismatch")
    truth_effect = np.stack([truth_map[task_id] for task_id in task_ids])
    source_map = load_npz_dict(assets / "SOURCE_GENE_EFFECTS.npz")
    source_effect = np.stack(
        [source_map[str(gene)] for gene in query.gene.astype(str)]
    )
    control_map = load_npz_dict(assets / "TARGET_CONTROL_PROFILES.npz")
    control = np.stack(
        [control_map[str(key)] for key in query.context_key.astype(str)]
    )
    (
        reference,
        source_control,
        mean_effect,
        train_gene_effect_map,
        guide_audit,
    ) = systema_reference(assets)
    query_genes = query.gene.astype(str).tolist()
    if not set(query_genes) <= set(train_gene_effect_map):
        raise ContractFailure(f"{setting}: missing train-only matching gene effect")
    train_matching_effect = np.stack(
        [train_gene_effect_map[gene] for gene in query_genes]
    )
    expected_shape = (int(spec["n_tasks"]), N_GENES)
    if (
        predictions.shape != (len(MODEL_KEYS), *expected_shape)
        or truth_effect.shape != expected_shape
        or source_effect.shape != expected_shape
        or train_matching_effect.shape != expected_shape
        or control.shape != expected_shape
        or not np.isfinite(predictions).all()
        or not np.isfinite(truth_effect).all()
        or not np.isfinite(source_effect).all()
        or not np.isfinite(train_matching_effect).all()
        or not np.isfinite(control).all()
    ):
        raise ContractFailure(f"{setting}: aligned formal arrays invalid")
    return {
        "setting": setting,
        "spec": spec,
        "query": query,
        "saved": saved,
        "genes": panel.gene_name.astype(str).to_numpy(),
        "predictions": predictions,
        "truth_effect": truth_effect,
        "source_effect": source_effect,
        "train_matching_effect": train_matching_effect,
        "control": control,
        "truth_post": control + truth_effect,
        "systema_source_train_reference": reference,
        "systema_source_mean_effect": mean_effect,
        "source_control": source_control,
        "guide_audit": guide_audit,
    }


def predictor_matrices(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    predictions = data["predictions"]
    control = data["control"]
    n_tasks = len(data["query"])
    source_train_reference = np.repeat(
        data["systema_source_train_reference"][None, :], n_tasks, axis=0
    )
    source_mean_effect = np.repeat(
        data["systema_source_mean_effect"][None, :], n_tasks, axis=0
    )
    result: dict[str, dict[str, Any]] = {}
    for index, key in enumerate(MODEL_KEYS):
        result[key] = {
            "family": "model_member",
            "effect": predictions[index],
            "post": control + predictions[index],
        }
    centroid_effect = predictions.mean(axis=0)
    result["family_centroid"] = {
        "family": "ensemble",
        "effect": centroid_effect,
        "post": control + centroid_effect,
    }
    result["matching_source_train_mean"] = {
        "family": "simple_baseline",
        "effect": data["train_matching_effect"],
        "post": control + data["train_matching_effect"],
    }
    result["matching_source_all_folds_mean"] = {
        "family": "simple_baseline",
        "effect": data["source_effect"],
        "post": control + data["source_effect"],
    }
    result["source_absolute_noncontrol_mean"] = {
        "family": "simple_baseline",
        "effect": source_train_reference - control,
        "post": source_train_reference,
    }
    result["target_control_plus_source_mean_effect"] = {
        "family": "simple_baseline",
        "effect": source_mean_effect,
        "post": control + source_mean_effect,
    }
    result["zero_effect"] = {
        "family": "simple_baseline",
        "effect": np.zeros_like(data["truth_effect"]),
        "post": control.copy(),
    }
    if set(result) != set(EXPECTED_PREDICTORS):
        raise ContractFailure("E197 predictor registry mismatch")
    return result


def predictor_input_audit(
    setting: str, predictors: dict[str, dict[str, Any]]
) -> pd.DataFrame:
    contracts = {
        "family_centroid": (
            "six frozen pretruth prediction releases",
            True,
            True,
        ),
        "matching_source_train_mean": (
            "Adamson train-only guide means + target control",
            True,
            False,
        ),
        "matching_source_all_folds_mean": (
            "Adamson all-fold source-gene asset + target control",
            True,
            True,
        ),
        "target_control_plus_source_mean_effect": (
            "Adamson train-only global mean effect + target control",
            True,
            False,
        ),
        "source_absolute_noncontrol_mean": (
            "Adamson train-only absolute perturbed centroid",
            False,
            False,
        ),
        "zero_effect": ("target control only", True, False),
    }
    rows = []
    for predictor in predictors:
        if predictor in MODEL_KEYS:
            contract = "frozen model-specific pretruth prediction release"
            uses_target_control = True
            uses_source_validation = True
        else:
            contract, uses_target_control, uses_source_validation = contracts[
                predictor
            ]
        rows.append(
            {
                "setting": setting,
                "predictor": predictor,
                "predictor_family": predictors[predictor]["family"],
                "input_contract": contract,
                "uses_target_control": uses_target_control,
                "uses_source_validation_or_all_folds": uses_source_validation,
                "uses_target_perturbed_expression_for_prediction": False,
                "role": (
                    "sensitivity_baseline"
                    if predictor == "matching_source_all_folds_mean"
                    else "primary_or_registered_comparator"
                ),
            }
        )
    return pd.DataFrame(rows)


def centroid_scores(
    predicted_post: np.ndarray,
    truth_post: np.ndarray,
    query: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    accuracy = np.full(len(query), np.nan, dtype=float)
    nearest = np.full(len(query), np.nan, dtype=float)
    competitors = np.zeros(len(query), dtype=int)
    batches = query.batch.astype(str).to_numpy()
    for batch in sorted(set(batches)):
        indices = np.flatnonzero(batches == batch)
        if len(indices) <= 1:
            continue
        candidates = truth_post[indices]
        for index in indices:
            distances = np.sqrt(
                np.sum(
                    (candidates - predicted_post[index][None, :]) ** 2,
                    axis=1,
                )
            )
            local = int(np.flatnonzero(indices == index)[0])
            correct = distances[local]
            other = np.delete(distances, local)
            accuracy[index] = float(np.mean(correct < other))
            nearest[index] = float(correct < np.min(other))
            competitors[index] = len(other)
    return accuracy, nearest, competitors


def gene_centroid_scores(
    data: dict[str, Any], predictors: dict[str, dict[str, Any]]
) -> pd.DataFrame:
    """Reconstruct one centroid per target gene in post and matched-effect spaces."""
    query = data["query"].reset_index(drop=True)
    gene_order = sorted(query.gene.astype(str).unique())
    indices_by_gene: dict[str, np.ndarray] = {}
    for gene in gene_order:
        indices = np.flatnonzero(query.gene.astype(str).to_numpy() == gene)
        weights = query.loc[indices, "n_target_cells"].to_numpy(dtype=np.float64)
        if len(indices) == 0 or np.any(weights <= 0):
            raise ContractFailure(f"malformed target gene centroid: {gene}")
        indices_by_gene[gene] = indices
    rows: list[dict[str, Any]] = []
    for predictor, item in predictors.items():
        for space, truth_matrix, predicted_matrix in (
            ("post_state", data["truth_post"], item["post"]),
            ("matched_control_effect", data["truth_effect"], item["effect"]),
        ):
            truth_centroids: list[np.ndarray] = []
            predicted_centroids: list[np.ndarray] = []
            for gene in gene_order:
                indices = indices_by_gene[gene]
                weights = query.loc[
                    indices, "n_target_cells"
                ].to_numpy(dtype=np.float64)
                truth_centroids.append(
                    np.average(truth_matrix[indices], axis=0, weights=weights)
                )
                predicted_centroids.append(
                    np.average(predicted_matrix[indices], axis=0, weights=weights)
                )
            truth_array = np.stack(truth_centroids)
            predicted_array = np.stack(predicted_centroids)
            distances = np.sqrt(
                np.maximum(
                    np.sum(
                        (predicted_array[:, None, :] - truth_array[None, :, :])
                        ** 2,
                        axis=2,
                    ),
                    0.0,
                )
            )
            for index, gene in enumerate(gene_order):
                correct = float(distances[index, index])
                other = np.delete(distances[index], index)
                indices = indices_by_gene[gene]
                rows.append(
                    {
                        "setting": data["setting"],
                        "predictor": predictor,
                        "predictor_family": item["family"],
                        "gene": gene,
                        "space": space,
                        "n_batches": int(len(indices)),
                        "n_target_cells": int(
                            query.loc[indices, "n_target_cells"].astype(int).sum()
                        ),
                        "systema_gene_centroid_accuracy": float(
                            np.mean(correct < other)
                        ),
                        "systema_gene_nearest_centroid_hit": float(
                            correct < np.min(other)
                        ),
                        "systema_n_competing_gene_centroids": int(len(other)),
                        "distance": "euclidean",
                        "aggregation": "target_cell_weighted_across_batches",
                        "ties": "miss",
                    }
                )
    return pd.DataFrame(rows)


def evaluate_systema(
    data: dict[str, Any],
    predictors: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    query = data["query"]
    truth_effect = data["truth_effect"]
    truth_post = data["truth_post"]
    source_train_reference = data["systema_source_train_reference"]
    source_mean_effect = data["systema_source_mean_effect"]
    genes = data["genes"]
    rows: list[dict[str, Any]] = []
    absolute_effect_top = [stable_top20(vector) for vector in truth_effect]
    source_top = [
        stable_top20(vector) for vector in data["train_matching_effect"]
    ]
    for predictor, item in predictors.items():
        effect = item["effect"]
        post = item["post"]
        ca, nearest, n_competitors = centroid_scores(post, truth_post, query)
        for index, task in enumerate(query.itertuples(index=False)):
            effect_pearson, effect_pearson_reason = safe_pearson(
                effect[index], truth_effect[index]
            )
            effect_cosine, effect_cosine_reason = safe_cosine(
                effect[index], truth_effect[index]
            )
            transport_pred = effect[index] - source_mean_effect
            transport_truth = truth_effect[index] - source_mean_effect
            transport_all, transport_all_reason = safe_pearson(
                transport_pred, transport_truth
            )
            proxy_indices = absolute_effect_top[index]
            transport_proxy, transport_proxy_reason = safe_pearson(
                transport_pred[proxy_indices], transport_truth[proxy_indices]
            )
            source_indices = source_top[index]
            transport_source, transport_source_reason = safe_pearson(
                transport_pred[source_indices], transport_truth[source_indices]
            )
            source_train_reference_pearson, source_train_reference_reason = safe_pearson(
                post[index] - source_train_reference,
                truth_post[index] - source_train_reference,
            )
            rows.append(
                {
                    "setting": data["setting"],
                    "predictor": predictor,
                    "predictor_family": item["family"],
                    "task_id": str(task.task_id),
                    "batch": str(task.batch),
                    "gene": str(task.gene),
                    "n_target_cells": int(task.n_target_cells),
                    "effect_pearson": effect_pearson,
                    "effect_pearson_na_reason": effect_pearson_reason,
                    "effect_cosine": effect_cosine,
                    "effect_cosine_na_reason": effect_cosine_reason,
                    "effect_rmse": rmse(effect[index], truth_effect[index]),
                    "post_rmse": rmse(post[index], truth_post[index]),
                    "systema_inspired_transport_pearson_delta_all": transport_all,
                    "systema_inspired_transport_pearson_delta_all_na_reason": (
                        transport_all_reason
                    ),
                    "systema_inspired_transport_pearson_delta_abs_effect_top20_proxy": (
                        transport_proxy
                    ),
                    "systema_inspired_transport_pearson_delta_abs_effect_top20_proxy_na_reason": (
                        transport_proxy_reason
                    ),
                    "systema_inspired_transport_pearson_delta_source_top20": transport_source,
                    "systema_inspired_transport_pearson_delta_source_top20_na_reason": (
                        transport_source_reason
                    ),
                    "systema_source_train_reference_pearson_delta_all": (
                        source_train_reference_pearson
                    ),
                    "systema_source_train_reference_pearson_delta_all_na_reason": (
                        source_train_reference_reason
                    ),
                    "absolute_effect_top20_proxy_genes": ";".join(
                        genes[proxy_indices]
                    ),
                    "source_top20_genes": ";".join(genes[source_indices]),
                    "systema_context_stratified_centroid_accuracy": ca[index],
                    "systema_context_stratified_nearest_centroid_hit": nearest[index],
                    "systema_n_competing_truth_centroids": int(
                        n_competitors[index]
                    ),
                    "systema_context_stratification": "within_target_batch",
                    "systema_centroid_distance": "euclidean_post_profile",
                    "systema_ties": "miss",
                }
            )
    return pd.DataFrame(rows)


def aggregate_by_gene(
    query: pd.DataFrame, matrix: np.ndarray
) -> tuple[list[str], np.ndarray, pd.DataFrame]:
    query = query.reset_index(drop=True)
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (len(query), N_GENES) or not np.isfinite(matrix).all():
        raise ContractFailure("gene aggregation received malformed matrix")
    gene_order = sorted(query.gene.astype(str).unique())
    vectors: list[np.ndarray] = []
    audit: list[dict[str, Any]] = []
    gene_values = query.gene.astype(str).to_numpy()
    for gene in gene_order:
        indices = np.flatnonzero(gene_values == gene)
        weights = query.loc[indices, "n_target_cells"].to_numpy(dtype=np.float64)
        if len(indices) == 0 or np.any(weights <= 0):
            raise ContractFailure(f"invalid gene aggregation weights: {gene}")
        vectors.append(np.average(matrix[indices], axis=0, weights=weights))
        audit.append(
            {
                "gene": gene,
                "n_batches": int(len(indices)),
                "n_target_cells": int(weights.sum()),
            }
        )
    return gene_order, np.stack(vectors), pd.DataFrame(audit)


def independent_rank_scores(
    truth: np.ndarray, prediction: np.ndarray, transpose: bool
) -> np.ndarray:
    prediction_norm = np.einsum("ij,ij->i", prediction, prediction)
    truth_norm = np.einsum("ij,ij->i", truth, truth)
    squared = np.maximum(
        prediction_norm[:, None]
        + truth_norm[None, :]
        - 2.0 * (prediction @ truth.T),
        0.0,
    )
    if transpose:
        squared = squared.T
    noise = np.random.default_rng(42).uniform(0, 1e-12, size=squared.shape)
    ranks = np.argsort(np.argsort(squared + noise, axis=0), axis=0)
    return np.diag(ranks).astype(np.float64) / max(len(truth) - 1, 1)


def run_scperteval(
    data: dict[str, Any],
    predictors: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    try:
        import anndata as ad
        import scperteval as sp
    except ImportError as exc:
        raise ContractFailure(
            "formal E197 must run in the frozen scperteval environment"
        ) from exc

    package_version = importlib.metadata.version("scperteval")
    module_path = Path(sp.__file__).resolve()
    try:
        module_path.relative_to(SCPERTEVAL_REPO.resolve())
    except ValueError as exc:
        raise ContractFailure(
            f"scPertEval imported outside frozen checkout: {module_path}"
        ) from exc
    query = data["query"].reset_index(drop=True)
    genes = data["genes"]
    gene_order, truth_by_gene, aggregation_audit = aggregate_by_gene(
        query, data["truth_effect"]
    )
    labels = ["control"] + gene_order
    obs_index = [f"pb_gt_{index}" for index in range(len(labels))]
    truth_x = np.vstack(
        [np.zeros((1, N_GENES), dtype=np.float64), truth_by_gene]
    ).astype(np.float32)
    truth_adata = ad.AnnData(
        X=truth_x,
        obs=pd.DataFrame({"perturbation": labels}, index=obs_index),
        var=pd.DataFrame(index=genes),
    )
    truth_adata.uns["E197_representation"] = (
        "one_observed_effect_centroid_per_unique_target_gene; not single cells"
    )

    prepared = {
        protocol: sp.prepare(
            truth_adata,
            protocol,
            min_cells=1,
            workers=1,
            seed=20260730,
            name=data["setting"],
        )
        for protocol in ALLOWED_SCPERTEVAL_PROTOCOLS
    }
    rows: list[pd.DataFrame] = []
    for predictor, item in predictors.items():
        observed_gene_order, prediction_by_gene, observed_audit = aggregate_by_gene(
            query, item["effect"]
        )
        if observed_gene_order != gene_order or not observed_audit.equals(
            aggregation_audit
        ):
            raise ContractFailure(
                f"{data['setting']} {predictor}: gene aggregation order changed"
            )
        pred_x = np.vstack(
            [np.zeros((1, N_GENES), dtype=np.float64), prediction_by_gene]
        ).astype(np.float32)
        truth_eval_float32 = truth_x[1:]
        truth_eval = truth_eval_float32.astype(np.float64)
        prediction_eval = pred_x[1:].astype(np.float64)
        prediction_adata = ad.AnnData(
            X=pred_x,
            obs=pd.DataFrame(
                {"perturbation": labels},
                index=[f"pb_pred_{predictor}_{i}" for i in range(len(labels))],
            ),
            var=pd.DataFrame(index=genes),
        )
        prediction_adata.uns["E197_representation"] = (
            "one_predicted_effect_centroid_per_unique_target_gene; not single cells"
        )
        truth_sum_float32 = truth_eval_float32.sum(axis=0)
        leave_one_out_reference = np.stack(
            [
                (
                    (truth_sum_float32 - truth_eval_float32[i])
                    / max(len(gene_order) - 1, 1)
                ).astype(np.float64)
                for i in range(len(gene_order))
            ]
        )
        manual_scores: dict[str, np.ndarray] = {
            "pearson": np.asarray(
                [
                    safe_pearson(prediction_eval[i], truth_eval[i])[0]
                    for i in range(len(gene_order))
                ]
            ),
            "pearson_ctrl": np.asarray(
                [
                    safe_pearson(prediction_eval[i], truth_eval[i])[0]
                    for i in range(len(gene_order))
                ]
            ),
            "pearson_pert": np.asarray(
                [
                    safe_pearson(
                        prediction_eval[i] - leave_one_out_reference[i],
                        truth_eval[i] - leave_one_out_reference[i],
                    )[0]
                    for i in range(len(gene_order))
                ]
            ),
            "mse": np.mean((prediction_eval - truth_eval) ** 2, axis=1),
            "rank": independent_rank_scores(
                truth_eval, prediction_eval, transpose=False
            ),
            "transpose_rank": independent_rank_scores(
                truth_eval, prediction_eval, transpose=True
            ),
        }
        for protocol in ALLOWED_SCPERTEVAL_PROTOCOLS:
            result = sp.score(prepared[protocol], protocol, prediction_adata)
            frame = result.per_perturbation.copy()
            if len(frame) != len(gene_order):
                raise ContractFailure(
                    f"{data['setting']} {predictor} {protocol}: "
                    "scPertEval task count mismatch"
                )
            score_map = dict(
                zip(
                    frame.perturbation.astype(str),
                    frame.score.to_numpy(dtype=float),
                )
            )
            ordered_scores = [score_map[gene] for gene in gene_order]
            rows.append(
                pd.DataFrame(
                    {
                        "setting": data["setting"],
                        "predictor": predictor,
                        "predictor_family": item["family"],
                        "gene": gene_order,
                        "n_batches": aggregation_audit.n_batches.astype(int),
                        "n_target_cells": (
                            aggregation_audit.n_target_cells.astype(int)
                        ),
                        "protocol": protocol,
                        "score": ordered_scores,
                        "independent_score": manual_scores[protocol],
                        "representation": "gene_level_effect_centroid",
                        "aggregation": "target_cell_weighted_across_batches",
                        "min_cells": 1,
                        "scperteval_version": package_version,
                        "scperteval_commit": SCPERTEVAL_COMMIT,
                        "scperteval_module": logical_path(module_path),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def family_endpoints(
    data: dict[str, Any],
    systema: pd.DataFrame,
) -> pd.DataFrame:
    query = data["query"].copy()
    saved = data["saved"]
    result = query[
        ["task_id", "batch", "gene", "n_target_cells", "context_key"]
    ].copy()
    result.insert(0, "setting", data["setting"])
    for column in (
        "diversity_lower_bound",
        "diameter_half_lower_bound",
        "predicted_magnitude",
        "source_effect_magnitude",
    ):
        result[column] = saved[column].to_numpy(dtype=float)

    member = systema[systema.predictor.isin(MODEL_KEYS)]
    centroid = systema[systema.predictor.eq("family_centroid")].set_index(
        "task_id"
    )
    for source_column, suffix in (
        (
            "systema_inspired_transport_pearson_delta_all",
            "inspired_transport_all",
        ),
        (
            "systema_inspired_transport_pearson_delta_abs_effect_top20_proxy",
            "inspired_transport_abs_effect_top20_proxy",
        ),
        (
            "systema_inspired_transport_pearson_delta_source_top20",
            "inspired_transport_source_top20",
        ),
        (
            "systema_source_train_reference_pearson_delta_all",
            "source_train_reference_all",
        ),
    ):
        member_error = (
            member.assign(error=1.0 - member[source_column])
            .groupby("task_id", observed=True)
            .error.mean()
        )
        result[
            f"family_member_mean_systema_pearson_error_{suffix}"
        ] = result.task_id.map(member_error)
        result[
            f"family_centroid_systema_pearson_error_{suffix}"
        ] = 1.0 - result.task_id.map(centroid[source_column])
    result["family_centroid_systema_context_stratified_centroid_error"] = (
        1.0
        - result.task_id.map(
            centroid.systema_context_stratified_centroid_accuracy
        )
    )

    member_mse = (
        member.assign(mse=member.effect_rmse.to_numpy(dtype=float) ** 2)
        .groupby("task_id", observed=True)
        .mse.mean()
    )
    result["family_member_mean_effect_mse"] = result.task_id.map(member_mse)
    result["family_centroid_effect_mse"] = result.task_id.map(
        centroid.effect_rmse.astype(float) ** 2
    )
    return result


def bootstrap_associations(
    endpoints: pd.DataFrame,
    estimand: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    risk_columns = (
        "diversity_lower_bound",
        "diameter_half_lower_bound",
        "predicted_magnitude",
        "source_effect_magnitude",
    )
    endpoint_columns = [
        column
        for column in endpoints.columns
        if column.startswith("family_")
    ]
    summaries: list[dict[str, Any]] = []
    draws: list[dict[str, Any]] = []
    for setting, frame in endpoints.groupby("setting", sort=True):
        frame = frame.reset_index(drop=True)
        for risk in risk_columns:
            for endpoint in endpoint_columns:
                keep = np.isfinite(frame[risk].to_numpy(float)) & np.isfinite(
                    frame[endpoint].to_numpy(float)
                )
                valid = frame.loc[keep].reset_index(drop=True)
                genes = np.asarray(sorted(valid.gene.astype(str).unique()))
                point = spearman(
                    valid[risk].to_numpy(float),
                    valid[endpoint].to_numpy(float),
                )
                seed = int(
                    hashlib.sha256(
                        f"E197\0{estimand}\0{setting}\0{risk}\0{endpoint}".encode()
                    ).hexdigest()[:8],
                    16,
                )
                values: list[float] = []
                eligible = len(valid) >= 20 and len(genes) >= 5
                if eligible:
                    groups = {
                        gene: np.flatnonzero(
                            valid.gene.astype(str).to_numpy() == gene
                        )
                        for gene in genes
                    }
                    rng = np.random.default_rng(seed)
                    x = valid[risk].to_numpy(float)
                    y = valid[endpoint].to_numpy(float)
                    for draw in range(N_BOOTSTRAP):
                        sampled = rng.choice(genes, size=len(genes), replace=True)
                        take = np.concatenate([groups[str(gene)] for gene in sampled])
                        value = spearman(x[take], y[take])
                        if math.isfinite(value):
                            values.append(value)
                            draws.append(
                                {
                                    "estimand": estimand,
                                    "setting": setting,
                                    "risk": risk,
                                    "endpoint": endpoint,
                                    "draw": draw,
                                    "spearman": value,
                                }
                            )
                summaries.append(
                    {
                        "estimand": estimand,
                        "setting": setting,
                        "risk": risk,
                        "endpoint": endpoint,
                        "spearman": point,
                        "ci95_lower": (
                            float(np.quantile(values, 0.025))
                            if values
                            else math.nan
                        ),
                        "ci95_upper": (
                            float(np.quantile(values, 0.975))
                            if values
                            else math.nan
                        ),
                        "bootstrap_valid": len(values),
                        "bootstrap_requested": N_BOOTSTRAP,
                        "bootstrap_unit": "target_gene",
                        "seed": seed,
                        "n_valid_tasks": len(valid),
                        "n_gene_clusters": len(genes),
                        "ci_eligible": eligible,
                    }
                )
    return pd.DataFrame(summaries), pd.DataFrame(draws)


def gene_equal_endpoints(endpoints: pd.DataFrame) -> pd.DataFrame:
    value_columns = [
        column
        for column in endpoints.columns
        if column
        in {
            "diversity_lower_bound",
            "diameter_half_lower_bound",
            "predicted_magnitude",
            "source_effect_magnitude",
        }
        or column.startswith("family_")
    ]
    grouped = (
        endpoints.groupby(["setting", "gene"], observed=True, sort=True)[
            value_columns
        ]
        .mean()
        .reset_index()
    )
    counts = (
        endpoints.groupby(["setting", "gene"], observed=True, sort=True)
        .size()
        .rename("n_batch_tasks")
        .reset_index()
    )
    return grouped.merge(
        counts, on=["setting", "gene"], validate="one_to_one"
    )


def context_centroid_eligibility(systema: pd.DataFrame) -> pd.DataFrame:
    family = systema[systema.predictor.eq("family_centroid")].copy()
    family["eligible"] = np.isfinite(
        family.systema_context_stratified_centroid_accuracy.to_numpy(float)
    )
    return (
        family.groupby(["setting", "batch"], observed=True, sort=True)
        .agg(
            n_gene_tasks=("task_id", "size"),
            n_eligible=("eligible", "sum"),
            n_competitors=("systema_n_competing_truth_centroids", "first"),
        )
        .reset_index()
    )


def formal_gates(
    systema: pd.DataFrame,
    scpert: pd.DataFrame,
    endpoints: pd.DataFrame,
    gene_centroids: pd.DataFrame,
    gene_endpoints: pd.DataFrame,
    input_audit: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for setting, spec in SETTINGS.items():
        n_tasks = int(spec["n_tasks"])
        n_predictors = N_PREDICTORS
        subset = systema[systema.setting.eq(setting)]
        gate(
            rows,
            setting,
            "systema_row_count",
            len(subset),
            n_tasks * n_predictors,
            len(subset) == n_tasks * n_predictors,
        )
        observed_predictors = sorted(subset.predictor.astype(str).unique())
        gate(
            rows,
            setting,
            "predictor_registry_exact",
            observed_predictors,
            sorted(EXPECTED_PREDICTORS),
            observed_predictors == sorted(EXPECTED_PREDICTORS),
        )
        gate(
            rows,
            setting,
            "systema_predictor_task_keys_unique",
            len(subset.drop_duplicates(["predictor", "task_id"])),
            len(subset),
            not subset.duplicated(["predictor", "task_id"]).any(),
        )
        for column in (
            "absolute_effect_top20_proxy_genes",
            "source_top20_genes",
        ):
            valid_top20 = subset[column].astype(str).map(
                lambda value: len(value.split(";")) == 20
                and len(set(value.split(";"))) == 20
            )
            gate(
                rows,
                setting,
                f"{column}_exactly_20_unique",
                int(valid_top20.sum()),
                len(subset),
                bool(valid_top20.all()),
            )
        source_consistency = (
            subset.groupby("task_id", observed=True).source_top20_genes.nunique()
        )
        source_gene_consistency = (
            subset.drop_duplicates(["task_id", "source_top20_genes"])
            .groupby("gene", observed=True)
            .source_top20_genes.nunique()
        )
        gate(
            rows,
            setting,
            "source_top20_predictor_and_batch_invariant",
            int((source_consistency.eq(1)).sum()),
            len(source_consistency),
            bool(source_consistency.eq(1).all())
            and bool(source_gene_consistency.eq(1).all()),
        )
        sc = scpert[scpert.setting.eq(setting)]
        n_genes = int(spec["n_genes"])
        gate(
            rows,
            setting,
            "scperteval_row_count",
            len(sc),
            n_genes * n_predictors * len(ALLOWED_SCPERTEVAL_PROTOCOLS),
            len(sc)
            == n_genes * n_predictors * len(ALLOWED_SCPERTEVAL_PROTOCOLS),
        )
        gate(
            rows,
            setting,
            "scperteval_predictor_gene_protocol_keys_unique",
            len(sc.drop_duplicates(["predictor", "gene", "protocol"])),
            len(sc),
            not sc.duplicated(["predictor", "gene", "protocol"]).any(),
        )
        observed_protocols = sorted(sc.protocol.astype(str).unique())
        gate(
            rows,
            setting,
            "scperteval_protocol_allowlist_exact",
            ",".join(observed_protocols),
            ",".join(sorted(ALLOWED_SCPERTEVAL_PROTOCOLS)),
            observed_protocols == sorted(ALLOWED_SCPERTEVAL_PROTOCOLS),
        )
        for protocol in ALLOWED_SCPERTEVAL_PROTOCOLS:
            block = sc[sc.protocol.eq(protocol)]
            score = block.score.to_numpy(dtype=float)
            independent = block.independent_score.to_numpy(dtype=float)
            finite_match = np.isfinite(score) == np.isfinite(independent)
            both_finite = np.isfinite(score) & np.isfinite(independent)
            max_delta = (
                float(np.max(np.abs(score[both_finite] - independent[both_finite])))
                if both_finite.any()
                else 0.0
            )
            gate(
                rows,
                setting,
                f"scperteval_{protocol}_matches_independent_formula",
                max_delta,
                "<=1e-8 and identical NA mask",
                bool(finite_match.all()) and max_delta <= 1e-8,
                "tolerance covers float32 centroid round-off only",
            )
        pearson = sc[sc.protocol.eq("pearson")].set_index(
            ["predictor", "gene"]
        ).score
        pearson_ctrl = sc[sc.protocol.eq("pearson_ctrl")].set_index(
            ["predictor", "gene"]
        ).score
        finite = np.isfinite(pearson) & np.isfinite(pearson_ctrl)
        max_p_delta = float(
            np.max(
                np.abs(
                    pearson.to_numpy(float)[finite]
                    - pearson_ctrl.to_numpy(float)[finite]
                )
            )
        )
        gate(
            rows,
            setting,
            "zero_control_pearson_equals_pearson_ctrl",
            max_p_delta,
            "<=1e-10",
            max_p_delta <= 1e-10,
        )
        e = endpoints[endpoints.setting.eq(setting)]
        gate(
            rows,
            setting,
            "family_endpoint_task_count",
            len(e),
            n_tasks,
            len(e) == n_tasks and e.task_id.astype(str).is_unique,
        )
        ge = gene_endpoints[gene_endpoints.setting.eq(setting)]
        gate(
            rows,
            setting,
            "gene_equal_endpoint_count",
            len(ge),
            n_genes,
            len(ge) == n_genes and ge.gene.astype(str).is_unique,
        )
        audit = input_audit[input_audit.setting.eq(setting)]
        gate(
            rows,
            setting,
            "predictor_input_audit_complete_and_pretruth",
            len(audit),
            n_predictors,
            len(audit) == n_predictors
            and set(audit.predictor.astype(str)) == set(EXPECTED_PREDICTORS)
            and not audit.uses_target_perturbed_expression_for_prediction.astype(
                bool
            ).any(),
        )
        gene = gene_centroids[gene_centroids.setting.eq(setting)]
        gate(
            rows,
            setting,
            "systema_gene_centroid_row_count",
            len(gene),
            int(spec["n_genes"]) * n_predictors * 2,
            len(gene) == int(spec["n_genes"]) * n_predictors * 2,
        )
        gate(
            rows,
            setting,
            "systema_gene_centroid_spaces",
            sorted(gene.space.astype(str).unique().tolist()),
            ["matched_control_effect", "post_state"],
            set(gene.space.astype(str))
            == {"matched_control_effect", "post_state"},
        )
        gate(
            rows,
            setting,
            "systema_gene_centroid_competitor_count",
            sorted(gene.systema_n_competing_gene_centroids.unique().tolist()),
            [int(spec["n_genes"]) - 1],
            set(gene.systema_n_competing_gene_centroids.astype(int))
            == {int(spec["n_genes"]) - 1},
        )
        context = subset[subset.predictor.eq("family_centroid")].copy()
        expected_competitors = (
            context.groupby("batch", observed=True).task_id.transform("size") - 1
        ).to_numpy(dtype=int)
        observed_competitors = (
            context.systema_n_competing_truth_centroids.to_numpy(dtype=int)
        )
        eligible = expected_competitors > 0
        observed_finite = np.isfinite(
            context.systema_context_stratified_centroid_accuracy.to_numpy(float)
        )
        gate(
            rows,
            setting,
            "context_centroid_eligibility_and_competitors",
            f"eligible={int(observed_finite.sum())};na={int((~observed_finite).sum())};"
            f"competitors={sorted(set(observed_competitors.tolist()))}",
            f"eligible={int(eligible.sum())};na={int((~eligible).sum())}",
            np.array_equal(observed_competitors, expected_competitors)
            and np.array_equal(observed_finite, eligible),
        )
    gate(
        rows,
        "global",
        "population_distribution_de_protocol_count",
        0,
        0,
        not scpert.protocol.astype(str).str.contains(
            "mmd|energy|sinkhorn|de_|wmse|top_k|degs", regex=True
        ).any(),
    )
    return pd.DataFrame(rows)


def predictor_summary(
    systema: pd.DataFrame,
    scpert: pd.DataFrame,
    gene_centroids: pd.DataFrame,
) -> pd.DataFrame:
    base = (
        systema.groupby(
            ["setting", "predictor", "predictor_family"],
            observed=True,
            sort=True,
        )
        .agg(
            n_tasks=("task_id", "size"),
            mean_effect_rmse=("effect_rmse", "mean"),
            mean_effect_pearson=("effect_pearson", "mean"),
            mean_systema_inspired_transport_pearson_delta_all=(
                "systema_inspired_transport_pearson_delta_all",
                "mean",
            ),
            mean_systema_inspired_transport_pearson_delta_abs_effect_top20_proxy=(
                "systema_inspired_transport_pearson_delta_abs_effect_top20_proxy",
                "mean",
            ),
            mean_systema_inspired_transport_pearson_delta_source_top20=(
                "systema_inspired_transport_pearson_delta_source_top20",
                "mean",
            ),
            mean_systema_source_train_reference_pearson_delta_all=(
                "systema_source_train_reference_pearson_delta_all",
                "mean",
            ),
            mean_systema_context_stratified_centroid_accuracy=(
                "systema_context_stratified_centroid_accuracy",
                "mean",
            ),
            mean_systema_context_stratified_nearest_hit=(
                "systema_context_stratified_nearest_centroid_hit",
                "mean",
            ),
        )
        .reset_index()
    )
    sc = (
        scpert.groupby(
            ["setting", "predictor", "predictor_family", "protocol"],
            observed=True,
            sort=True,
        )
        .score.mean()
        .unstack("protocol")
        .add_prefix("mean_scperteval_")
        .reset_index()
    )
    keys = ["setting", "predictor", "predictor_family"]
    gene_summaries: list[pd.DataFrame] = []
    for space, suffix in (
        ("matched_control_effect", "effect"),
        ("post_state", "post"),
    ):
        block = (
            gene_centroids[gene_centroids.space.eq(space)]
            .groupby(keys, observed=True, sort=True)
            .agg(
                **{
                    f"n_gene_centroids_{suffix}": ("gene", "size"),
                    f"mean_systema_gene_centroid_accuracy_{suffix}": (
                        "systema_gene_centroid_accuracy",
                        "mean",
                    ),
                    f"mean_systema_gene_nearest_hit_{suffix}": (
                        "systema_gene_nearest_centroid_hit",
                        "mean",
                    ),
                }
            )
            .reset_index()
        )
        gene_summaries.append(block)
    gene = gene_summaries[0].merge(
        gene_summaries[1], on=keys, validate="one_to_one"
    )
    return base.merge(
        sc,
        on=keys,
        validate="one_to_one",
    ).merge(
        gene,
        on=keys,
        validate="one_to_one",
    )


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]

    def render(value: Any) -> str:
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4f}"
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines)


def make_figures(
    summary: pd.DataFrame, associations: pd.DataFrame
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    predictor_order = [
        "family_centroid",
        "matching_source_train_mean",
        "matching_source_all_folds_mean",
        "target_control_plus_source_mean_effect",
        "source_absolute_noncontrol_mean",
        "zero_effect",
    ]
    labels = [
        "Family centroid",
        "Matching train",
        "Matching all folds",
        "Transported mean",
        "Source absolute mean",
        "Zero effect",
    ]
    colors = [
        "#3B6FB6",
        "#4E9A73",
        "#79B79A",
        "#C8873A",
        "#9B77B4",
        "#8B8B8B",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.1))
    measures = [
        (
            "mean_systema_inspired_transport_pearson_delta_all",
            "A  Systema-inspired transported Pearson-Δ",
            "Higher is better",
        ),
        (
            "mean_systema_inspired_transport_pearson_delta_abs_effect_top20_proxy",
            "B  Pearson-Δ (absolute-effect top20 proxy)",
            "Higher is better",
        ),
        (
            "mean_systema_gene_centroid_accuracy_effect",
            "C  Selected-gene centroid accuracy (effect space)",
            "Higher is better",
        ),
        (
            "mean_scperteval_mse",
            "D  scPertEval pseudobulk MSE",
            "Lower is better",
        ),
    ]
    settings = list(SETTINGS)
    width = 0.36
    x = np.arange(len(predictor_order))
    for axis, (column, title, ylabel) in zip(axes.flat, measures):
        for offset, setting in enumerate(settings):
            block = summary[
                summary.setting.eq(setting)
                & summary.predictor.isin(predictor_order)
            ].set_index("predictor")
            values = [block.loc[predictor, column] for predictor in predictor_order]
            axis.bar(
                x + (offset - 0.5) * width,
                values,
                width,
                color=colors,
                alpha=1.0 if offset == 0 else 0.62,
                edgecolor="#333333",
                linewidth=0.35,
                label=setting.replace("_", " "),
            )
        axis.set_xticks(x, labels, rotation=20, ha="right")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#E4E4E4", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "E197 · Post-truth centroid-only evaluation",
        x=0.06,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = FIGURES / "E197_systema_scperteval_summary"
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    endpoint_order = [
        "family_member_mean_systema_pearson_error_inspired_transport_all",
        "family_member_mean_systema_pearson_error_inspired_transport_abs_effect_top20_proxy",
        "family_member_mean_systema_pearson_error_inspired_transport_source_top20",
        "family_member_mean_effect_mse",
    ]
    risk_order = [
        "diversity_lower_bound",
        "diameter_half_lower_bound",
        "predicted_magnitude",
        "source_effect_magnitude",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.1, 4.1))
    for axis, setting in zip(axes, settings):
        block = associations[
            associations.estimand.eq("gene_equal")
            & associations.setting.eq(setting)
            & associations.endpoint.isin(endpoint_order)
            & associations.risk.isin(risk_order)
        ].pivot(index="risk", columns="endpoint", values="spearman")
        matrix = block.reindex(index=risk_order, columns=endpoint_order).to_numpy()
        image = axis.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix[row, col]
                axis.text(
                    col,
                    row,
                    "NA" if not np.isfinite(value) else f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if np.isfinite(value) and abs(value) > 0.48 else "#222222",
                    fontsize=8,
                )
        axis.set_xticks(
            range(4), ["Transport all", "Effect top20 proxy", "Source top20", "MSE"]
        )
        axis.set_yticks(
            range(4), ["Diversity", "Diameter / 2", "Pred. magnitude", "Source magnitude"]
        )
        axis.tick_params(axis="x", rotation=24)
        axis.set_title(setting.replace("_", " "), loc="left", fontweight="bold")
        axis.spines[:].set_visible(False)
    fig.colorbar(image, ax=axes, fraction=0.028, pad=0.04, label="Spearman ρ")
    fig.suptitle(
        "E197 · Risk ordering changes with the error definition",
        x=0.06,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.16, right=0.90, bottom=0.23, top=0.82, wspace=0.28)
    path = FIGURES / "E197_risk_endpoint_heatmap"
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_report(
    summary: pd.DataFrame,
    associations: pd.DataFrame,
    gates: pd.DataFrame,
    guide_audit: pd.DataFrame,
) -> None:
    focus = summary[
        summary.predictor.isin(
            [
                "family_centroid",
                "matching_source_train_mean",
                "matching_source_all_folds_mean",
                "target_control_plus_source_mean_effect",
                "source_absolute_noncontrol_mean",
                "zero_effect",
            ]
        )
    ][
        [
            "setting",
            "predictor",
            "mean_systema_inspired_transport_pearson_delta_all",
            "mean_systema_inspired_transport_pearson_delta_abs_effect_top20_proxy",
            "mean_systema_gene_centroid_accuracy_effect",
            "mean_systema_gene_centroid_accuracy_post",
            "mean_scperteval_mse",
            "mean_scperteval_rank",
        ]
    ]
    best_rows = []
    for setting in SETTINGS:
        block = associations[
            associations.estimand.eq("gene_equal")
            & associations.setting.eq(setting)
            & associations.endpoint.eq(
                "family_member_mean_systema_pearson_error_inspired_transport_all"
            )
        ].sort_values("spearman", ascending=False)
        if len(block):
            best_rows.append(block.iloc[0])
    best = pd.DataFrame(best_rows)[
        ["setting", "risk", "spearman", "ci95_lower", "ci95_upper"]
    ]
    guide_counts = ", ".join(
        f"{setting}={block.guide.nunique()}"
        for setting, block in guide_audit.groupby("setting", sort=True)
    )
    lines = [
        "# E197｜Systema 与 scPertEval 均值层评价",
        "",
        "## 定位",
        "",
        "本分析使用 E190/E192 已开封的旧结果，标签是 "
        "`POSTTRUTH_EXPLORATORY`。它补齐多指标评价，不是新的盲法确认。",
        "",
        "现有预测只保存任务均值，因此 scPertEval 先按目标细胞数把 batch×gene "
        "效应合并为每个 target gene 一个 centroid，再运行 pseudobulk/centroid "
        "协议。没有运行 MMD、Energy、Sinkhorn、DE-AUPRC、DE-AUROC、"
        "DE-overlap、WMSE，也没有复制均值伪造预测细胞。",
        "",
        "## 主要预测器与简单基线",
        "",
        markdown_table(focus),
        "",
        "跨数据集主 Pearson-Δ 使用 `target control + source mean effect` 作为"
        "Systema-inspired 训练侧参考；另存 source train absolute reference "
        "敏感性结果，不把前者称为 Systema 官方原式。基因 centroid "
        "accuracy 按目标细胞数重建每个基因的真实与预测质心，再使用 Systema "
        "官方欧氏距离公式。主列先减去匹配 target control；post-state 列保留批次"
        "构成敏感性。MSE 与 rank 越低越好。",
        "",
        "## 既有风险量对 Systema 全基因误差的排序",
        "",
        markdown_table(best),
        "",
        "表中点估计先在每个 target gene 内平均各 batch，再让 47/21 个基因等权；"
        "区间按 target gene bootstrap 5,000 次。task-weighted 结果另存于完整表。"
        "这些相关性只说明旧数据中的排序关系，不会把 E190/E192 的 "
        "ABSTAIN/PASS gate 重新判一次。",
        "",
        "## Systema reference 与 top20 边界",
        "",
        f"Adamson train guide 数：{guide_counts}。每个 guide 的四个 train 伪重复先"
        "按细胞数合并，再对 guide 等权。`matching_source_train_mean` 只读 train；"
        "`matching_source_all_folds_mean` 读取既有 source all-fold asset，只作为更强"
        "敏感性基线。两者与全局 source mean 分开保存。",
        "",
        "目标文件只有任务均值，没有逐细胞差异检验。因此 `abs-effect top20` 是按"
        "真实效应绝对值选出的事后代理，不称为官方 Systema Pearson-Δ20；"
        "source top20 完全由 Adamson 训练效应确定。",
        "",
        "## 完整性",
        "",
        f"- formal gates：{int(gates.passed.astype(bool).sum())}/{len(gates)} 通过；",
        f"- scPertEval official source commit：`{SCPERTEVAL_COMMIT}`；",
        f"- Systema official source commit：`{SYSTEMA_COMMIT}`；",
        "- 图均为白底 PNG/PDF；完整 task 指标、官方协议原始分数和 bootstrap "
        "结果见 `tables/`。",
        "",
        "## 解释边界",
        "",
        "单个 pseudobulk centroid 不能恢复细胞内异质性或实验重复性。本结果不能写成"
        "完整 scPertEval population benchmark，也不能用来保证投稿录用。",
    ]
    write_text_atomic(REPORTS / "E197_REPORT.md", "\n".join(lines) + "\n")
    write_text_atomic(
        OUT / "README_先看这个.md",
        "# E197 先看这个\n\n"
        "先读 `reports/E197_REPORT.md`；指标冻结见 `ANALYSIS_FREEZE.md`。"
        "本实验是 E190/E192 开真值后的探索性多指标审计。\n"
    )


def verify_prepared_hashes() -> None:
    path = TABLES / "E197_INPUT_HASHES.csv"
    if not path.is_file():
        raise ContractFailure("run prepare and commit the freeze before formal analysis")
    frozen = pd.read_csv(path, keep_default_na=False)
    current_hashes, current_gates = collect_prepare()
    compare_columns = ["role", "path", "bytes", "sha256"]
    if not frozen[compare_columns].equals(current_hashes[compare_columns]):
        raise ContractFailure("prepared E197 input hashes changed")
    if not current_gates.passed.astype(bool).all():
        raise ContractFailure("prepared E197 metadata gate changed")
    required_tracked = [
        SCRIPT.relative_to(ROOT).as_posix(),
        FREEZE.relative_to(ROOT).as_posix(),
        path.relative_to(ROOT).as_posix(),
        (TABLES / "E197_PREPARE_GATES.csv").relative_to(ROOT).as_posix(),
        (OUT / "E197_STATUS.json").relative_to(ROOT).as_posix(),
    ]
    try:
        git_output(ROOT, "ls-files", "--error-unmatch", *required_tracked)
    except subprocess.CalledProcessError as exc:
        raise ContractFailure(
            "E197 freeze/runner/prepare artifacts must be committed before formal run"
        ) from exc
    dirty = git_output(ROOT, "status", "--short", "--", *required_tracked)
    if dirty:
        raise ContractFailure(
            "E197 committed freeze/runner/prepare artifacts changed before formal run"
        )
    head = git_output(ROOT, "rev-parse", "HEAD")
    branch = git_output(ROOT, "branch", "--show-current")
    if not branch:
        raise ContractFailure("formal E197 requires a named git branch")
    for remote in ("origin", "github"):
        try:
            remote_tip = git_output(ROOT, "rev-parse", f"{remote}/{branch}")
        except subprocess.CalledProcessError as exc:
            raise ContractFailure(
                f"formal E197 requires a fetched {remote}/{branch} tip"
            ) from exc
        if remote_tip != head:
            raise ContractFailure(
                f"formal E197 requires HEAD pushed to {remote}: {head} != {remote_tip}"
            )


def _analyze_formal() -> dict[str, Any]:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    systema_frames: list[pd.DataFrame] = []
    scpert_frames: list[pd.DataFrame] = []
    endpoint_frames: list[pd.DataFrame] = []
    gene_centroid_frames: list[pd.DataFrame] = []
    guide_frames: list[pd.DataFrame] = []
    input_audit_frames: list[pd.DataFrame] = []
    for setting, spec in SETTINGS.items():
        data = load_setting(setting, spec)
        predictors = predictor_matrices(data)
        input_audit_frames.append(predictor_input_audit(setting, predictors))
        systema = evaluate_systema(data, predictors)
        gene_centroids = gene_centroid_scores(data, predictors)
        scpert = run_scperteval(data, predictors)
        endpoints = family_endpoints(data, systema)
        systema_frames.append(systema)
        scpert_frames.append(scpert)
        endpoint_frames.append(endpoints)
        gene_centroid_frames.append(gene_centroids)
        guide = data["guide_audit"].copy()
        guide.insert(0, "setting", setting)
        guide_frames.append(guide)
    systema = pd.concat(systema_frames, ignore_index=True)
    scpert = pd.concat(scpert_frames, ignore_index=True)
    endpoints = pd.concat(endpoint_frames, ignore_index=True)
    gene_centroids = pd.concat(gene_centroid_frames, ignore_index=True)
    guide_audit = pd.concat(guide_frames, ignore_index=True)
    input_audit = pd.concat(input_audit_frames, ignore_index=True)
    gene_endpoints = gene_equal_endpoints(endpoints)
    context_eligibility = context_centroid_eligibility(systema)
    gates = formal_gates(
        systema,
        scpert,
        endpoints,
        gene_centroids,
        gene_endpoints,
        input_audit,
    )
    task_associations, task_draws = bootstrap_associations(
        endpoints, "task_weighted_gene_cluster_bootstrap"
    )
    gene_associations, gene_draws = bootstrap_associations(
        gene_endpoints, "gene_equal"
    )
    associations = pd.concat(
        [task_associations, gene_associations], ignore_index=True
    )
    draws = pd.concat([task_draws, gene_draws], ignore_index=True)
    summary = predictor_summary(systema, scpert, gene_centroids)

    systema.to_csv(TABLES / "E197_SYSTEMA_TASK_METRICS.csv", index=False)
    scpert.to_csv(TABLES / "E197_SCPERTEVAL_PSEUDOBULK_SCORES.csv", index=False)
    endpoints.to_csv(TABLES / "E197_FAMILY_TASK_ENDPOINTS.csv", index=False)
    gene_centroids.to_csv(
        TABLES / "E197_SYSTEMA_GENE_CENTROID_METRICS.csv", index=False
    )
    gene_endpoints.to_csv(
        TABLES / "E197_GENE_EQUAL_ENDPOINTS.csv", index=False
    )
    summary.to_csv(TABLES / "E197_PREDICTOR_SUMMARY.csv", index=False)
    associations.to_csv(
        TABLES / "E197_GENE_CLUSTER_BOOTSTRAP_SUMMARY.csv", index=False
    )
    draws.to_csv(
        TABLES / "E197_GENE_CLUSTER_BOOTSTRAP_DRAWS.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    guide_audit.to_csv(TABLES / "E197_SYSTEMA_GUIDE_AUDIT.csv", index=False)
    input_audit.to_csv(TABLES / "E197_PREDICTOR_INPUT_AUDIT.csv", index=False)
    context_eligibility.to_csv(
        TABLES / "E197_CONTEXT_CENTROID_ELIGIBILITY.csv", index=False
    )
    gates.to_csv(TABLES / "E197_FORMAL_GATES.csv", index=False)
    make_figures(summary, associations)
    write_report(summary, associations, gates, guide_audit)
    return {
        "formal_git_head": git_output(ROOT, "rev-parse", "HEAD"),
        "n_settings": len(SETTINGS),
        "n_tasks": int(endpoints.task_id.nunique()),
        "n_setting_tasks": int(len(endpoints)),
        "n_predictors": N_PREDICTORS,
        "n_systema_task_rows": int(len(systema)),
        "n_scperteval_rows": int(len(scpert)),
        "n_systema_gene_centroid_rows": int(len(gene_centroids)),
        "scperteval_protocols": list(ALLOWED_SCPERTEVAL_PROTOCOLS),
        "scperteval_commit": SCPERTEVAL_COMMIT,
        "systema_commit": SYSTEMA_COMMIT,
        "bootstrap_unit": "target_gene",
        "bootstrap_draws": N_BOOTSTRAP,
        "association_estimands": [
            "task_weighted_gene_cluster_bootstrap",
            "gene_equal",
        ],
        "formal_gates_passed": int(gates.passed.astype(bool).sum()),
        "formal_gates_total": int(len(gates)),
        "population_or_de_protocols_run": False,
        "prediction_cells_fabricated": False,
        "changes_E190_or_E192_gate": False,
    }


def analyze(allow_posttruth: bool) -> None:
    if not allow_posttruth:
        raise ContractFailure(
            "formal analysis is post-truth; pass --allow-posttruth-evaluation"
        )
    status_path = OUT / "E197_STATUS.json"
    if status_path.is_file():
        previous = json.loads(status_path.read_text())
        if str(previous.get("status", "")).startswith("COMPLETE"):
            raise ContractFailure("E197 is already COMPLETE; formal rerun refused")
    if OUTPUT_HASH_INDEX.exists():
        raise ContractFailure("E197 output hash index already exists; formal rerun refused")
    formal_payloads = [
        TABLES / "E197_SYSTEMA_TASK_METRICS.csv",
        TABLES / "E197_SYSTEMA_GENE_CENTROID_METRICS.csv",
        TABLES / "E197_GENE_EQUAL_ENDPOINTS.csv",
        TABLES / "E197_SCPERTEVAL_PSEUDOBULK_SCORES.csv",
        TABLES / "E197_FAMILY_TASK_ENDPOINTS.csv",
        TABLES / "E197_PREDICTOR_SUMMARY.csv",
        TABLES / "E197_GENE_CLUSTER_BOOTSTRAP_SUMMARY.csv",
        TABLES / "E197_GENE_CLUSTER_BOOTSTRAP_DRAWS.csv.gz",
        TABLES / "E197_SYSTEMA_GUIDE_AUDIT.csv",
        TABLES / "E197_PREDICTOR_INPUT_AUDIT.csv",
        TABLES / "E197_CONTEXT_CENTROID_ELIGIBILITY.csv",
        TABLES / "E197_FORMAL_GATES.csv",
        FIGURES / "E197_systema_scperteval_summary.png",
        FIGURES / "E197_systema_scperteval_summary.pdf",
        FIGURES / "E197_risk_endpoint_heatmap.png",
        FIGURES / "E197_risk_endpoint_heatmap.pdf",
        REPORTS / "E197_REPORT.md",
        OUT / "README_先看这个.md",
    ]
    existing_payloads = [logical_path(path) for path in formal_payloads if path.exists()]
    if existing_payloads:
        raise ContractFailure(
            "formal output payloads already exist; archive or remove the failed run "
            f"before retry: {existing_payloads}"
        )
    verify_prepared_hashes()
    write_status(
        "RUNNING_POSTTRUTH_EXPLORATORY",
        formal_run_acknowledged=True,
        population_or_de_protocols_run=False,
    )
    try:
        fields = _analyze_formal()
        write_status("COMPLETE_POSTTRUTH_EXPLORATORY", **fields)
        final_paths = formal_payloads + [OUT / "E197_STATUS.json"]
        hashes = output_hashes(final_paths)
        print(
            json.dumps(
                {
                    "status": "COMPLETE_POSTTRUTH_EXPLORATORY",
                    **fields,
                    "output_hashes": int(len(hashes)),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    except Exception as exc:
        write_status(
            "FAILED_POSTTRUTH_EXPLORATORY",
            error_type=type(exc).__name__,
            error_message=str(exc),
            population_or_de_protocols_run=False,
        )
        raise


def synthetic_smoke() -> None:
    """Exercise formulas, cluster bootstrap and official centroid protocols."""
    rng = np.random.default_rng(197)
    n_tasks = 12
    truth = rng.normal(size=(n_tasks, N_GENES))
    prediction = truth.copy()
    reference = rng.normal(size=N_GENES)
    for index in range(n_tasks):
        value, reason = safe_pearson(
            prediction[index] - reference, truth[index] - reference
        )
        if reason or abs(value - 1.0) > 1e-12:
            raise ContractFailure("synthetic Systema Pearson identity failed")
    query = pd.DataFrame(
        {
            "batch": ["b0"] * 6 + ["b1"] * 6,
            "gene": [f"g{i}" for i in range(6)] * 2,
        }
    )
    accuracy, nearest, competitors = centroid_scores(prediction, truth, query)
    if (
        not np.allclose(accuracy, 1.0)
        or not np.allclose(nearest, 1.0)
        or set(competitors) != {5}
    ):
        raise ContractFailure("synthetic centroid accuracy identity failed")
    query["n_target_cells"] = np.arange(1, n_tasks + 1)
    gene_identity = gene_centroid_scores(
        {
            "setting": "synthetic",
            "query": query,
            "truth_post": truth,
            "truth_effect": truth,
        },
        {
            "perfect": {
                "family": "synthetic",
                "post": truth.copy(),
                "effect": truth.copy(),
            }
        },
    )
    if (
        set(gene_identity.space) != {"matched_control_effect", "post_state"}
        or not np.allclose(gene_identity.systema_gene_centroid_accuracy, 1.0)
        or not np.allclose(gene_identity.systema_gene_nearest_centroid_hit, 1.0)
    ):
        raise ContractFailure("synthetic gene-centroid identity failed")

    try:
        import anndata as ad
        import scperteval as sp
    except ImportError as exc:
        raise ContractFailure(
            "synthetic smoke requires frozen scperteval environment"
        ) from exc
    genes = np.asarray([f"gene_{i}" for i in range(N_GENES)])
    labels = ["control"] + [f"task_{i}" for i in range(n_tasks)]
    matrix = np.vstack([np.zeros((1, N_GENES)), truth]).astype(np.float32)
    ground = ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(
            {"perturbation": labels}, index=[f"g{i}" for i in range(len(labels))]
        ),
        var=pd.DataFrame(index=genes),
    )
    predicted = ad.AnnData(
        X=matrix.copy(),
        obs=pd.DataFrame(
            {"perturbation": labels}, index=[f"p{i}" for i in range(len(labels))]
        ),
        var=pd.DataFrame(index=genes),
    )
    protocol_results = {}
    for protocol in ALLOWED_SCPERTEVAL_PROTOCOLS:
        prepared = sp.prepare(
            ground,
            protocol,
            min_cells=1,
            workers=1,
            seed=20260730,
            name="E197_synthetic",
        )
        result = sp.score(prepared, protocol, predicted)
        values = result.per_perturbation.score.to_numpy(float)
        expected = 0.0 if protocol in {"mse", "rank", "transpose_rank"} else 1.0
        if not np.allclose(values, expected, atol=1e-8):
            raise ContractFailure(f"synthetic scPertEval failed: {protocol}")
        protocol_results[protocol] = float(np.mean(values))

    noisy_matrix = matrix.copy()
    noisy_matrix[1:] += rng.normal(0.0, 0.35, size=truth.shape).astype(np.float32)
    noisy = ad.AnnData(
        X=noisy_matrix,
        obs=pd.DataFrame(
            {"perturbation": labels},
            index=[f"n{i}" for i in range(len(labels))],
        ),
        var=pd.DataFrame(index=genes),
    )
    truth_eval_float32 = matrix[1:]
    truth_eval = truth_eval_float32.astype(np.float64)
    noisy_eval = noisy_matrix[1:].astype(np.float64)
    truth_sum_float32 = truth_eval_float32.sum(axis=0)
    leave_one_out_reference = np.stack(
        [
            (
                (truth_sum_float32 - truth_eval_float32[i])
                / (n_tasks - 1)
            ).astype(np.float64)
            for i in range(n_tasks)
        ]
    )
    independent = {
        "pearson": np.asarray(
            [safe_pearson(noisy_eval[i], truth_eval[i])[0] for i in range(n_tasks)]
        ),
        "pearson_ctrl": np.asarray(
            [safe_pearson(noisy_eval[i], truth_eval[i])[0] for i in range(n_tasks)]
        ),
        "pearson_pert": np.asarray(
            [
                safe_pearson(
                    noisy_eval[i] - leave_one_out_reference[i],
                    truth_eval[i] - leave_one_out_reference[i],
                )[0]
                for i in range(n_tasks)
            ]
        ),
        "mse": np.mean((noisy_eval - truth_eval) ** 2, axis=1),
        "rank": independent_rank_scores(
            truth_eval, noisy_eval, transpose=False
        ),
        "transpose_rank": independent_rank_scores(
            truth_eval, noisy_eval, transpose=True
        ),
    }
    noisy_deltas: dict[str, float] = {}
    for protocol in ALLOWED_SCPERTEVAL_PROTOCOLS:
        prepared = sp.prepare(
            ground,
            protocol,
            min_cells=1,
            workers=1,
            seed=20260730,
            name=f"E197_synthetic_noisy_{protocol}",
        )
        result = sp.score(prepared, protocol, noisy)
        score_map = dict(
            zip(
                result.per_perturbation.perturbation.astype(str),
                result.per_perturbation.score.to_numpy(float),
            )
        )
        observed = np.asarray([score_map[label] for label in labels[1:]])
        expected = independent[protocol]
        finite_mask_matches = np.array_equal(
            np.isfinite(observed), np.isfinite(expected)
        )
        finite = np.isfinite(observed) & np.isfinite(expected)
        max_delta = (
            float(np.max(np.abs(observed[finite] - expected[finite])))
            if finite.any()
            else 0.0
        )
        if not finite_mask_matches or max_delta > 1e-8:
            raise ContractFailure(
                f"synthetic independent scPertEval mismatch: {protocol} "
                f"max_delta={max_delta}"
            )
        noisy_deltas[protocol] = max_delta

    cluster_frame = pd.DataFrame(
        {
            "setting": ["synthetic"] * 25,
            "task_id": [f"t{i}" for i in range(25)],
            "gene": np.repeat([f"cluster_{i}" for i in range(5)], 5),
            "diversity_lower_bound": np.arange(25, dtype=float),
            "diameter_half_lower_bound": np.arange(25, dtype=float) * 2.0,
            "predicted_magnitude": np.arange(25, dtype=float) * 3.0,
            "source_effect_magnitude": np.arange(25, dtype=float) * 4.0,
            "family_synthetic_error": np.arange(25, dtype=float) + 0.5,
        }
    )
    cluster_summary, cluster_draws = bootstrap_associations(
        cluster_frame, "synthetic_task_weighted"
    )
    if (
        len(cluster_summary) != 4
        or not (cluster_summary.bootstrap_valid == N_BOOTSTRAP).all()
        or not np.allclose(cluster_summary.spearman, 1.0)
        or len(cluster_draws) != 4 * N_BOOTSTRAP
    ):
        raise ContractFailure("synthetic gene-cluster bootstrap failed")
    print(
        json.dumps(
            {
                "status": "PASS",
                "uses_only_synthetic_data": True,
                "formal_prediction_or_truth_array_payloads_loaded": False,
                "systema_identity": True,
                "centroid_accuracy_identity": True,
                "gene_centroid_identity_both_spaces": True,
                "gene_cluster_bootstrap": True,
                "scperteval_protocol_results": protocol_results,
                "scperteval_independent_formula_max_delta": noisy_deltas,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("prepare", "analyze", "synthetic-smoke"),
        default="prepare",
    )
    parser.add_argument("--allow-posttruth-evaluation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        prepare()
    elif args.mode == "synthetic-smoke":
        synthetic_smoke()
    else:
        analyze(args.allow_posttruth_evaluation)


if __name__ == "__main__":
    main()
