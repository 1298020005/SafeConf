#!/usr/bin/env python3
"""E158: one-shot prospective evaluation of locked PRESCRIBE P3/P4 runs.

The ordering is intentional: both E157 checkpoints and both truth-free task
score tables are byte-verified before this program touches the raw Norman H5AD.
The first access is recorded in an append-only unseal event.  There is no CLI
surface for changing the frozen tasks, metrics, bootstrap count, or coverage.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import rankdata
from sklearn.utils.extmath import safe_sparse_dot


ROOT = Path(__file__).resolve().parents[2]
RAW = Path(
    "/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/"
    "NormanWeissman2019_filtered.h5ad"
)
E155 = ROOT / "docs/实验结果/E155_prescribe_norman_p3p4_contract_20260714"
E156 = ROOT / "docs/实验结果/E156_prescribe_norman_p3p4_preprocess_20260714"
E157 = ROOT / "docs/实验结果/E157_prescribe_norman_p3p4_native_20260714"
BASE_OUT = ROOT / "docs/实验结果/E158_prescribe_norman_p3p4_forward_evaluation_20260714"
# OUT is selected once in main as the first eligible append-only attempt directory.
OUT = BASE_OUT
CONTRACT = BASE_OUT / "ANALYSIS_CONTRACT.md"

EXPECTED_RAW_SHA256 = "efde6f5301fe256725dce1d980f37bd96a13481a9a16135515897368e631affc"
EXPECTED_E157_PHASE = "complete_checkpoint_and_label_only_scores_locked_no_test_truth_access"
EXPECTED_E156_PHASE = "complete_preprocessing_only_no_training_no_evaluation"
SEED = 3407
N_BOOT = 10_000
N_PCA = 10
N_GENES = 2_044
MIN_VALID_BOOT = 9_500
COVERAGES = np.round(np.arange(0.50, 1.001, 0.05), 2)
FOCAL_COVERAGES = (0.90, 0.95)

PANELS = {
    "Norman_P3": {
        "slug": "norman_p3",
        "run": E157 / "norman_p3_formal_seed3407",
    },
    "Norman_P4": {
        "slug": "norman_p4",
        "run": E157 / "norman_p4_formal_seed3407",
    },
}

SCORES = {
    "combined_confidence": "combined_confidence_official",
    "predicted_magnitude": "predicted_magnitude_rms",
}

TARGETS = {
    "pearson_effect_accuracy": {"favorable_sign": 1.0, "role": "primary"},
    "frac_correct_direction_all": {"favorable_sign": 1.0, "role": "secondary"},
    "frac_correct_direction_top20_de": {"favorable_sign": 1.0, "role": "supplementary"},
    "rmse_effect_error": {"favorable_sign": -1.0, "role": "secondary"},
    "raw_pearson_effect_accuracy_sensitivity": {"favorable_sign": 1.0, "role": "sensitivity"},
    "raw_frac_correct_direction_all_sensitivity": {"favorable_sign": 1.0, "role": "sensitivity"},
    "raw_rmse_effect_error_sensitivity": {"favorable_sign": -1.0, "role": "sensitivity"},
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def reject_preunseal_raw_alias(path: Path, role: str) -> None:
    """Use metadata only to reject symlink/hardlink aliases of the raw test file."""
    candidate = path.expanduser()
    absolute = candidate.absolute()
    if candidate.is_symlink() or any(parent.is_symlink() for parent in absolute.parents):
        raise RuntimeError(f"{role} uses a symlink before test unseal: {candidate}")
    raw_resolved = RAW.expanduser().resolve()
    if candidate.resolve() == raw_resolved:
        raise RuntimeError(f"{role} aliases the raw test file before unseal")
    if candidate.exists() and RAW.exists() and os.path.samefile(candidate, RAW):
        raise RuntimeError(f"{role} is a hardlink alias of the raw test file before unseal")


def sha256_file(path: Path, *, allow_raw: bool = False) -> str:
    if not allow_raw:
        reject_preunseal_raw_alias(path, f"SHA256 target {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def fixed_seed(label: str) -> int:
    return int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def update_status(path: Path, **updates: object) -> dict[str, object]:
    current: dict[str, object] = {}
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
    current.update(updates)
    atomic_json(path, current)
    return current


def committed_provenance() -> dict[str, object]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    result: dict[str, object] = {"git_head_at_unseal": head}
    for role, path in {"runner": Path(__file__).resolve(), "contract": CONTRACT.resolve()}.items():
        relative = path.relative_to(ROOT)
        try:
            committed = subprocess.check_output(
                ["git", "show", f"HEAD:{relative.as_posix()}"], cwd=ROOT
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"E158 {role} is not committed at Git HEAD") from exc
        committed_hash = hashlib.sha256(committed).hexdigest()
        working_hash = sha256_file(path)
        if committed_hash != working_hash:
            raise RuntimeError(f"E158 {role} differs from its Git HEAD blob")
        result[f"{role}_sha256"] = working_hash
        result[f"{role}_matches_git_head_blob"] = True
    return result


def normalize_condition(value: str) -> str:
    parts = str(value).replace("control", "ctrl").split("_")
    if len(parts) == 1 and parts[0] == "ctrl":
        return "ctrl"
    if len(parts) == 1:
        parts.append("ctrl")
    return "+".join(sorted(parts))


def require_file_hash(
    path: Path, expected: str, role: str, *, allow_raw: bool = False
) -> str:
    if not allow_raw:
        reject_preunseal_raw_alias(path, role)
    if not path.is_file():
        raise FileNotFoundError(f"{role}: {path}")
    observed = sha256_file(path, allow_raw=allow_raw)
    if observed != expected:
        raise RuntimeError(f"{role} SHA256 changed: {observed} != {expected}")
    return observed


def require_exact_path(path: Path, expected: Path, role: str) -> Path:
    """Reject status-provided paths before opening or hashing their targets."""
    resolved = path.expanduser().resolve()
    expected_resolved = expected.expanduser().resolve()
    if resolved != expected_resolved:
        raise RuntimeError(f"{role} path changed: {resolved} != {expected_resolved}")
    return resolved


def require_direct_child(path: Path, directory: Path, suffix: str, role: str) -> Path:
    """Allow a generated filename only as a direct child of one frozen directory."""
    resolved = path.expanduser().resolve()
    parent = directory.expanduser().resolve()
    if resolved.parent != parent or resolved.suffix != suffix:
        raise RuntimeError(f"{role} path is outside the frozen allowlist: {resolved}")
    return resolved


def require_git_blob_match(path: Path, role: str) -> str:
    """Require a repository asset to be byte-identical to the current HEAD blob."""
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError(f"{role} is not a repository asset: {resolved}") from exc
    try:
        committed = subprocess.check_output(
            ["git", "show", f"HEAD:{relative.as_posix()}"], cwd=ROOT
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"{role} is not committed at Git HEAD") from exc
    committed_hash = hashlib.sha256(committed).hexdigest()
    working_hash = sha256_file(resolved)
    if committed_hash != working_hash:
        raise RuntimeError(f"{role} differs from its Git HEAD blob")
    return working_hash


def expected_tasks(panel: str, e155_status: dict[str, object]) -> tuple[list[str], str]:
    path = E155 / "manifests" / f"{panel}_SPLIT.csv"
    relative = f"manifests/{panel}_SPLIT.csv"
    require_git_blob_match(path, f"{panel} E155 split")
    require_file_hash(path, str(e155_status["artifact_sha256"][relative]), relative)
    frame = pd.read_csv(path)
    tasks = frame.loc[frame["split"].eq("test"), "condition"].astype(str).tolist()
    if len(tasks) != 24 or len(set(tasks)) != 24:
        raise RuntimeError(f"{panel}: E155 does not contain 24 unique test tasks")
    return tasks, sha256_file(path)


def verify_e157_panel(
    panel: str,
    config: dict[str, object],
    expected: list[str],
    e156_status: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object], list[dict[str, str]]]:
    run = Path(config["run"])
    status_path = run / "STATUS.json"
    if not status_path.is_file():
        raise FileNotFoundError(f"{panel}: formal E157 STATUS is absent: {status_path}")
    require_git_blob_match(status_path, f"{panel} E157 status")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    required_equal = {
        "phase": EXPECTED_E157_PHASE,
        "mode": "formal",
        "seed": SEED,
        "panel": panel,
    }
    for key, value in required_equal.items():
        if status.get(key) != value:
            raise RuntimeError(f"{panel}: E157 {key}={status.get(key)!r}, expected {value!r}")
    if status.get("test_expression_accessed") is not False:
        raise RuntimeError(f"{panel}: E157 does not certify test_expression_accessed=false")
    if status.get("test_endpoint_computed") is not False:
        raise RuntimeError(f"{panel}: E157 already reports a test endpoint")
    if status.get("runner_matches_git_head_blob") is not True:
        raise RuntimeError(f"{panel}: E157 runner was not frozen at its training HEAD")
    if status.get("contract_matches_git_head_blob") is not True:
        raise RuntimeError(f"{panel}: E157 contract was not frozen at its training HEAD")

    runner = ROOT / "tools/scripts/run_e157_prescribe_norman_p3p4_native.py"
    require_file_hash(runner, str(status["runner_sha256"]), f"{panel} E157 runner")
    require_file_hash(
        E157 / "ANALYSIS_CONTRACT.md",
        str(status["contract_sha256"]),
        f"{panel} E157 contract",
    )

    source_manifest = run / "SOURCE_MANIFEST.csv"
    input_manifest = run / "INPUT_MANIFEST.csv"
    require_git_blob_match(source_manifest, f"{panel} E157 source manifest")
    require_file_hash(
        source_manifest, str(status["source_manifest_sha256"]), f"{panel} source manifest"
    )
    require_git_blob_match(input_manifest, f"{panel} E157 input manifest")
    require_file_hash(
        input_manifest, str(status["input_manifest_sha256"]), f"{panel} input manifest"
    )

    forward = run / "E157_DEVELOPMENT_FORWARD_EQUIVALENCE.csv"
    require_git_blob_match(forward, f"{panel} E157 forward equivalence")
    require_file_hash(
        forward,
        str(status["development_forward_equivalence_sha256"]),
        f"{panel} forward equivalence",
    )
    forward_max_delta = float(status["development_forward_max_abs_delta"])
    if not np.isfinite(forward_max_delta):
        raise RuntimeError(f"{panel}: label-only forward equivalence is non-finite")
    if forward_max_delta > 1e-5:
        raise RuntimeError(f"{panel}: label-only forward equivalence exceeds 1e-5")
    graph_audit_path = run / "DEVELOPMENT_GRAPH_AUDIT.csv"
    require_git_blob_match(graph_audit_path, f"{panel} development graph audit")
    require_file_hash(
        graph_audit_path,
        str(status["development_graph_audit_sha256"]),
        f"{panel} development graph audit",
    )

    checkpoint_records = []
    checkpoint_audit = status.get("checkpoint_audit", {})
    best_val_loss = float(checkpoint_audit.get("best_val_loss", np.nan))
    if not np.isfinite(best_val_loss):
        raise RuntimeError(f"{panel}: E157 best validation loss is absent or non-finite")
    for key in ["best_lightning_checkpoint", "locked_slim_checkpoint"]:
        record = checkpoint_audit.get(key)
        if not isinstance(record, dict):
            raise RuntimeError(f"{panel}: missing E157 checkpoint audit {key}")
        if key == "best_lightning_checkpoint":
            path = require_direct_child(
                Path(str(record["path"])),
                run / "checkpoints/main",
                ".ckpt",
                f"{panel} {key}",
            )
        else:
            path = require_exact_path(
                Path(str(record["path"])),
                Path("/home/yyf/data/safeconf_e157_locked_models")
                / str(config["slug"])
                / "E157_LOCKED_NATIVE_STATE.pt",
                f"{panel} {key}",
            )
        observed = require_file_hash(path, str(record["sha256"]), f"{panel} {key}")
        if path.stat().st_size != int(record["bytes"]):
            raise RuntimeError(f"{panel}: {key} byte size changed")
        if record.get("contains_anndata") is not False:
            raise RuntimeError(f"{panel}: {key} contains AnnData")
        if record.get("contains_test_expression_path") is not False:
            raise RuntimeError(f"{panel}: {key} contains a test-expression path")
        checkpoint_records.append({"role": key, "path": str(path), "sha256": observed})

    score_audit = status.get("locked_task_score_audit")
    if not isinstance(score_audit, dict):
        raise RuntimeError(f"{panel}: E157 locked task-score audit is absent")
    score_path = require_exact_path(
        Path(str(score_audit["path"])),
        run / "locked/E157_LOCKED_LABEL_ONLY_TASK_SCORES.csv",
        f"{panel} locked task scores",
    )
    require_git_blob_match(score_path, f"{panel} E157 locked task scores")
    require_file_hash(score_path, str(score_audit["sha256"]), f"{panel} locked task scores")
    if int(score_audit.get("n_tasks", -1)) != 24:
        raise RuntimeError(f"{panel}: E157 score audit does not certify 24 tasks")
    if score_audit.get("contains_test_truth") is not False:
        raise RuntimeError(f"{panel}: E157 score audit does not certify absence of test truth")
    if score_audit.get("contains_test_expression") is not False:
        raise RuntimeError(f"{panel}: E157 score audit does not certify absence of test expression")
    scores = pd.read_csv(score_path)
    required_columns = {
        "panel",
        "task_id",
        "query_has_test_expression",
        "query_has_y",
        "query_has_y_pca",
        "log_prob",
        "epistemic_confidence",
        "aleatoric_confidence",
        "combined_confidence_official",
        "predicted_magnitude_rms",
        "gene_order_sha256",
        *{f"predicted_pca_{index}" for index in range(N_PCA)},
    }
    missing = required_columns - set(scores.columns)
    if missing:
        raise RuntimeError(f"{panel}: locked task score columns missing: {sorted(missing)}")
    if len(scores) != 24 or scores["task_id"].duplicated().any():
        raise RuntimeError(f"{panel}: locked task table is not 24 unique tasks")
    if set(scores["task_id"].astype(str)) != set(expected):
        raise RuntimeError(f"{panel}: locked tasks differ from E155")
    if not scores["panel"].astype(str).eq(panel).all():
        raise RuntimeError(f"{panel}: panel label changed in locked task table")
    for field in ["query_has_test_expression", "query_has_y", "query_has_y_pca"]:
        if not scores[field].eq(False).all():
            raise RuntimeError(f"{panel}: locked label-only query violates {field}=false")
    numeric_columns = [
        "log_prob",
        "epistemic_confidence",
        "aleatoric_confidence",
        "combined_confidence_official",
        "predicted_magnitude_rms",
        *[f"predicted_pca_{index}" for index in range(N_PCA)],
    ]
    for column in numeric_columns:
        scores[column] = pd.to_numeric(scores[column], errors="raise")
    numeric = scores[numeric_columns].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise RuntimeError(f"{panel}: non-finite locked score")
    recombined = 2.0 * scores["epistemic_confidence"].to_numpy(float) + scores[
        "aleatoric_confidence"
    ].to_numpy(float)
    if not np.allclose(
        recombined, scores["combined_confidence_official"].to_numpy(float), rtol=1e-10, atol=1e-12
    ):
        raise RuntimeError(f"{panel}: official combined-confidence formula changed")

    panel_records = [item for item in e156_status["panels"] if item["panel"] == panel]
    if len(panel_records) != 1:
        raise RuntimeError(f"{panel}: E156 panel record is not unique")
    dev_h5ad = require_exact_path(
        Path(str(panel_records[0]["h5ad"])),
        Path("/home/yyf/data/safeconf_e156_prescribe")
        / str(config["slug"])
        / "perturb_processed.h5ad",
        f"{panel} E156 dev H5AD",
    )
    require_file_hash(
        dev_h5ad,
        str(e156_status["perturb_processed_sha256"][panel]),
        f"{panel} E156 dev H5AD",
    )
    if str(status["e156_dev_h5ad_sha256"]) != str(
        e156_status["perturb_processed_sha256"][panel]
    ):
        raise RuntimeError(f"{panel}: E157 recorded a different E156 dev H5AD")
    if str(status["e156_cell_graphs_sha256"]) != str(e156_status["cell_graphs_sha256"][panel]):
        raise RuntimeError(f"{panel}: E157 recorded a different E156 graph cache")
    graph_records = [
        item for item in e156_status["graph_summaries"] if item["panel"] == panel
    ]
    if len(graph_records) != 1:
        raise RuntimeError(f"{panel}: E156 graph record is not unique")
    graph_path = require_exact_path(
        Path(str(graph_records[0]["cell_graphs_path"])),
        Path("/home/yyf/data/safeconf_e156_prescribe")
        / str(config["slug"])
        / "data_pyg/cell_graphs.pkl",
        f"{panel} E156 cell graph cache",
    )
    graph_hash = require_file_hash(
        graph_path,
        str(e156_status["cell_graphs_sha256"][panel]),
        f"{panel} E156 cell graph cache",
    )

    records = [
        {"role": "E157_status", "path": str(status_path), "sha256": sha256_file(status_path)},
        {"role": "E157_source_manifest", "path": str(source_manifest), "sha256": sha256_file(source_manifest)},
        {"role": "E157_input_manifest", "path": str(input_manifest), "sha256": sha256_file(input_manifest)},
        {"role": "E157_forward_equivalence", "path": str(forward), "sha256": sha256_file(forward)},
        {"role": "E157_development_graph_audit", "path": str(graph_audit_path), "sha256": sha256_file(graph_audit_path)},
        {"role": "E157_locked_task_scores", "path": str(score_path), "sha256": sha256_file(score_path)},
        {"role": "E156_dev_h5ad", "path": str(dev_h5ad), "sha256": sha256_file(dev_h5ad)},
        {"role": "E156_cell_graphs", "path": str(graph_path), "sha256": graph_hash},
        *checkpoint_records,
    ]
    panel_audit = {
        "panel": panel,
        "status_path": str(status_path),
        "status_sha256": sha256_file(status_path),
        "checkpoint_sha256": {
            item["role"]: item["sha256"] for item in checkpoint_records
        },
        "task_score_path": str(score_path),
        "task_score_sha256": sha256_file(score_path),
        "n_locked_tasks": int(len(scores)),
        "best_val_loss": best_val_loss,
        "test_expression_accessed": False,
        "test_endpoint_computed": False,
    }
    return scores, panel_audit, records


def verify_all_locked_inputs() -> tuple[
    dict[str, pd.DataFrame], dict[str, object], pd.DataFrame, pd.DataFrame
]:
    if not CONTRACT.is_file():
        raise FileNotFoundError(CONTRACT)
    provenance = committed_provenance()
    e155_status_path = E155 / "RUN_STATUS.json"
    e156_status_path = E156 / "RUN_STATUS.json"
    require_git_blob_match(e155_status_path, "E155 status")
    require_git_blob_match(e156_status_path, "E156 status")
    e155_status = json.loads(e155_status_path.read_text(encoding="utf-8"))
    e156_status = json.loads(e156_status_path.read_text(encoding="utf-8"))
    if e155_status.get("phase") != "contract_frozen_predictions_and_errors_unseen":
        raise RuntimeError("E155 is not in the frozen-unseen phase")
    if e156_status.get("phase") != EXPECTED_E156_PHASE:
        raise RuntimeError("E156 is not strict-preprocessing complete")
    for key in ["test_X_rows_indexed", "test_X_rows_materialized", "test_X_rows_transformed"]:
        if e156_status.get(key) is not False:
            raise RuntimeError(f"E156 does not certify {key}=false")
    if e156_status.get("test_expression_transformed") is not False:
        raise RuntimeError("E156 already transformed test expression")
    artifact_manifest = E156 / "tables/E156_ARTIFACT_HASHES.csv"
    require_git_blob_match(artifact_manifest, "E156 artifact manifest")
    require_file_hash(
        artifact_manifest,
        str(e156_status["artifact_manifest_sha256"]),
        "E156 artifact manifest",
    )

    locked_tables: dict[str, pd.DataFrame] = {}
    panel_audits: dict[str, object] = {}
    manifest_rows: list[dict[str, str]] = [
        {"role": "E155_status", "path": str(e155_status_path), "sha256": sha256_file(e155_status_path)},
        {"role": "E156_status", "path": str(e156_status_path), "sha256": sha256_file(e156_status_path)},
        {"role": "E156_artifact_manifest", "path": str(artifact_manifest), "sha256": sha256_file(artifact_manifest)},
        {"role": "E158_runner", "path": str(Path(__file__).resolve()), "sha256": str(provenance["runner_sha256"])},
        {"role": "E158_contract", "path": str(CONTRACT), "sha256": str(provenance["contract_sha256"])},
    ]
    task_sets: dict[str, set[str]] = {}
    for panel, config in PANELS.items():
        tasks, split_hash = expected_tasks(panel, e155_status)
        task_sets[panel] = set(tasks)
        manifest_rows.append(
            {
                "role": "E155_split",
                "path": str(E155 / "manifests" / f"{panel}_SPLIT.csv"),
                "sha256": split_hash,
            }
        )
        scores, audit, records = verify_e157_panel(panel, config, tasks, e156_status)
        scores = scores.copy()
        scores["locked_row_order"] = np.arange(len(scores), dtype=int)
        locked_tables[panel] = scores
        panel_audits[panel] = audit
        manifest_rows.extend(records)
    if task_sets["Norman_P3"] & task_sets["Norman_P4"]:
        raise RuntimeError("P3/P4 locked test tasks overlap")
    expected_union = task_sets["Norman_P3"] | task_sets["Norman_P4"]
    if len(expected_union) != 48:
        raise RuntimeError("P3/P4 do not define exactly 48 disjoint test tasks")
    condition_path = E155 / "manifests/E155_CONDITION_AUDIT.csv"
    condition_role = "manifests/E155_CONDITION_AUDIT.csv"
    require_git_blob_match(condition_path, "E155 condition audit")
    condition_hash = require_file_hash(
        condition_path,
        str(e155_status["artifact_sha256"][condition_role]),
        "E155 condition audit",
    )
    condition_audit = pd.read_csv(condition_path)
    if condition_audit["condition"].isna().any() or condition_audit["condition"].duplicated().any():
        raise RuntimeError("E155 condition audit does not have unique non-null conditions")
    indexed_tasks = set(condition_audit["condition"].astype(str))
    if not expected_union.issubset(indexed_tasks):
        raise RuntimeError("E155 condition audit does not cover all 48 locked test tasks")
    frozen_counts = pd.to_numeric(
        condition_audit.set_index("condition").loc[sorted(expected_union), "n_cells_obs_metadata"],
        errors="raise",
    ).to_numpy(float)
    if not np.isfinite(frozen_counts).all() or np.any(frozen_counts <= 0) or not np.equal(
        frozen_counts, np.floor(frozen_counts)
    ).all():
        raise RuntimeError("E155 condition audit has invalid frozen test-cell counts")
    manifest_rows.append(
        {"role": "E155_condition_audit", "path": str(condition_path), "sha256": condition_hash}
    )
    return (
        locked_tables,
        {**provenance, "panels": panel_audits},
        pd.DataFrame(manifest_rows),
        condition_audit.set_index("condition"),
    )


def write_once(path: Path, payload: bytes) -> str:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return sha256_file(path)


def fixed_log_normalize(matrix: sp.spmatrix | np.ndarray) -> sp.csr_matrix:
    counts = sp.csr_matrix(matrix, dtype=np.float32)
    totals = np.asarray(counts.sum(axis=1)).reshape(-1)
    if np.any(~np.isfinite(totals)) or np.any(totals <= 0):
        raise RuntimeError("Test input contains a zero/non-finite library-size cell")
    normalized = sp.diags(np.asarray(10_000.0 / totals, np.float32), format="csr") @ counts
    normalized = normalized.tocsr().astype(np.float32)
    np.log1p(normalized.data, out=normalized.data)
    normalized.eliminate_zeros()
    return normalized


def pca_transform(matrix: sp.spmatrix, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    transformed = safe_sparse_dot(matrix, components.T, dense_output=True)
    transformed -= np.asarray(mean @ components.T, dtype=np.float32)[None, :]
    return np.asarray(transformed, dtype=np.float32)


def load_frozen_transform() -> dict[str, object]:
    model_path = Path("/home/yyf/data/safeconf_e156_prescribe/shared_train_fit/TRAIN_ONLY_PCA_MODEL.npz")
    artifact_manifest = pd.read_csv(E156 / "tables/E156_ARTIFACT_HASHES.csv")
    expected_rows = artifact_manifest.loc[
        artifact_manifest["path"].astype(str).eq(str(model_path)), "sha256"
    ]
    if len(expected_rows) != 1:
        raise RuntimeError("E156 artifact manifest does not uniquely identify TRAIN_ONLY_PCA_MODEL.npz")
    model_hash = require_file_hash(
        model_path, str(expected_rows.iloc[0]), "E156 frozen train-only PCA model"
    )
    with np.load(model_path, allow_pickle=False) as model:
        genes = model["model_genes"].astype(str).tolist()
        mean = np.asarray(model["mean"], np.float32)
        components = np.asarray(model["components"], np.float32)
    if len(genes) != N_GENES or mean.shape != (N_GENES,) or components.shape != (N_PCA, N_GENES):
        raise RuntimeError("Frozen E156 PCA model has an unexpected shape")
    if len(set(genes)) != N_GENES:
        raise RuntimeError("Frozen E156 gene axis is not unique")
    if not np.isfinite(mean).all() or not np.isfinite(components).all():
        raise RuntimeError("Frozen E156 PCA model contains non-finite values")

    controls = []
    for panel in PANELS:
        dev = Path(
            f"/home/yyf/data/safeconf_e156_prescribe/{PANELS[panel]['slug']}/perturb_processed.h5ad"
        )
        adata = sc.read_h5ad(dev, backed="r")
        try:
            if adata.var_names.astype(str).tolist() != genes:
                raise RuntimeError(f"{panel}: E156 gene order differs from shared PCA model")
            if not np.array_equal(np.asarray(adata.uns["pca_mean"], np.float32), mean):
                raise RuntimeError(f"{panel}: embedded PCA mean differs from frozen model")
            if not np.array_equal(np.asarray(adata.uns["pca_components"], np.float32), components):
                raise RuntimeError(f"{panel}: embedded PCA components differ from frozen model")
            ctrl_mask = adata.obs["condition"].astype(str).eq("ctrl").to_numpy()
            if int(ctrl_mask.sum()) <= 0:
                raise RuntimeError(f"{panel}: E156 dev H5AD has no shared-train control cells")
            control = np.asarray(adata[ctrl_mask].to_memory().X.mean(axis=0)).reshape(-1)
            if control.shape != (N_GENES,):
                raise RuntimeError(f"{panel}: shared-train control has an unexpected shape")
            controls.append(np.asarray(control, np.float32))
        finally:
            adata.file.close()
    if not np.allclose(controls[0], controls[1], rtol=0.0, atol=1e-7):
        raise RuntimeError("P3/P4 do not have the same frozen train-control mean")
    if not np.isfinite(controls[0]).all():
        raise RuntimeError("Frozen shared-train control contains non-finite values")
    return {
        "model_path": model_path,
        "model_sha256": model_hash,
        "genes": genes,
        "gene_order_sha256": sha256_text("\n".join(genes)),
        "mean": mean,
        "components": components,
        "control": controls[0],
        "control_sha256": hashlib.sha256(controls[0].tobytes()).hexdigest(),
    }


def verify_locked_score_semantics(
    locked_tables: dict[str, pd.DataFrame], transform: dict[str, object]
) -> dict[str, object]:
    """Reproduce every truth-free score semantic before authorizing raw access."""
    expected_gene_hash = str(transform["gene_order_sha256"])
    components = np.asarray(transform["components"], float)
    mean = np.asarray(transform["mean"], float)
    control = np.asarray(transform["control"], float)
    panel_audit: dict[str, object] = {}
    for panel, scores in locked_tables.items():
        if not scores["gene_order_sha256"].astype(str).eq(expected_gene_hash).all():
            raise RuntimeError(f"{panel}: E157 score gene-order hash differs from E156")
        predicted_pca = scores[
            [f"predicted_pca_{index}" for index in range(N_PCA)]
        ].to_numpy(float)
        reconstructed = predicted_pca @ components + mean[None, :]
        magnitude = np.sqrt(np.mean((reconstructed - control[None, :]) ** 2, axis=1))
        locked_magnitude = scores["predicted_magnitude_rms"].to_numpy(float)
        delta = np.abs(magnitude - locked_magnitude)
        if not np.isfinite(delta).all() or not np.allclose(
            magnitude, locked_magnitude, rtol=1e-6, atol=1e-7
        ):
            raise RuntimeError(f"{panel}: locked predicted magnitude does not reproduce pre-unseal")
        panel_audit[panel] = {
            "n_tasks": int(len(scores)),
            "gene_order_sha256": expected_gene_hash,
            "predicted_magnitude_max_abs_delta": float(delta.max()),
            "verified_without_test_truth_or_expression": True,
        }
    return panel_audit


def vector_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    value = np.corrcoef(x, y)[0, 1]
    return float(value) if np.isfinite(value) else float("nan")


def safe_correlation(x: np.ndarray, y: np.ndarray, method: str) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x, float)[mask], np.asarray(y, float)[mask]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    if method == "spearman":
        x, y = rankdata(x), rankdata(y)
    elif method != "pearson":
        raise ValueError(method)
    return vector_pearson(x, y)


def unseal_and_compute_tasks(
    locked_tables: dict[str, pd.DataFrame],
    transform: dict[str, object],
    condition_audit: pd.DataFrame,
    status_path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    all_tasks = set().union(*[set(table["task_id"].astype(str)) for table in locked_tables.values()])

    update_status(
        status_path,
        phase="raw_container_open_started_after_hash_and_unseal",
        raw_container_open_started=True,
        raw_container_opened=False,
        raw_obs_access_started=False,
        raw_obs_read=False,
        test_X_access_started=False,
        test_X_rows_materialized=False,
    )
    adata = sc.read_h5ad(RAW, backed="r")
    try:
        update_status(
            status_path,
            phase="raw_obs_access_started_after_container_open",
            raw_container_opened=True,
            raw_obs_access_started=True,
            raw_obs_read=False,
        )
        normalized_labels = adata.obs["perturbation"].astype(str).map(normalize_condition)
        update_status(
            status_path,
            phase="raw_obs_read_fixed_task_index_validation",
            raw_obs_read=True,
        )
        observed = set(normalized_labels[normalized_labels.isin(all_tasks)].astype(str))
        if observed != all_tasks:
            raise RuntimeError(f"Raw Norman tasks differ: missing={sorted(all_tasks-observed)}")
        if not set(transform["genes"]).issubset(set(adata.var_names.astype(str))):
            raise RuntimeError("Raw Norman data lacks a frozen E156 model gene")
        mask = normalized_labels.isin(all_tasks).to_numpy()
        update_status(
            status_path,
            phase="test_X_access_started_after_both_E157_locks",
            raw_container_opened=True,
            raw_obs_read=True,
            test_X_access_started=True,
            test_X_rows_materialized=False,
        )
        subset = adata[mask, list(transform["genes"])].to_memory()
        labels = normalized_labels.loc[mask].astype(str).to_numpy()
    finally:
        adata.file.close()
    update_status(
        status_path,
        phase="test_X_rows_materialized_fixed_transform_started",
        test_X_rows_materialized=True,
        n_test_cells_materialized=int(len(labels)),
    )
    subset_genes = subset.var_names.astype(str).tolist()
    if (
        len(subset_genes) != N_GENES
        or len(set(subset_genes)) != N_GENES
        or subset_genes != list(transform["genes"])
    ):
        raise RuntimeError("Materialized test matrix does not preserve the frozen 2,044-gene order")

    matrix = fixed_log_normalize(subset.X)
    pca = pca_transform(matrix, np.asarray(transform["mean"]), np.asarray(transform["components"]))
    control = np.asarray(transform["control"], float)
    components = np.asarray(transform["components"], float)
    mean = np.asarray(transform["mean"], float)
    rows = []
    count_audit = []
    for panel, scores in locked_tables.items():
        expected_gene_hash = str(transform["gene_order_sha256"])
        if not scores["gene_order_sha256"].astype(str).eq(expected_gene_hash).all():
            raise RuntimeError(f"{panel}: E157 score gene-order hash differs from E156")
        for record in scores.to_dict("records"):
            task = str(record["task_id"])
            task_mask = labels == task
            n_cells = int(task_mask.sum())
            expected_cells = int(condition_audit.loc[task, "n_cells_obs_metadata"])
            if n_cells != expected_cells:
                raise RuntimeError(
                    f"{panel}/{task}: test cell count {n_cells} != frozen metadata {expected_cells}"
                )
            truth_pca_mean = np.asarray(pca[task_mask].mean(axis=0), float)
            truth_reconstructed = truth_pca_mean @ components + mean
            raw_truth_mean = np.asarray(matrix[task_mask].mean(axis=0)).reshape(-1).astype(float)
            predicted_pca = np.asarray(
                [record[f"predicted_pca_{index}"] for index in range(N_PCA)], float
            )
            predicted_reconstructed = predicted_pca @ components + mean
            pred_effect = predicted_reconstructed - control
            truth_effect = truth_reconstructed - control
            raw_truth_effect = raw_truth_mean - control
            magnitude = float(np.sqrt(np.mean(pred_effect**2)))
            if not np.isclose(
                magnitude, float(record["predicted_magnitude_rms"]), rtol=1e-6, atol=1e-7
            ):
                raise RuntimeError(f"{panel}/{task}: locked predicted magnitude does not reproduce")
            top20 = np.argsort(-np.abs(truth_effect), kind="stable")[:20]
            rows.append(
                {
                    "panel": panel,
                    "task_id": task,
                    "locked_row_order": int(record["locked_row_order"]),
                    "n_test_cells": n_cells,
                    "n_genes": N_GENES,
                    "gene_order_sha256": expected_gene_hash,
                    "epistemic_confidence": float(record["epistemic_confidence"]),
                    "aleatoric_confidence": float(record["aleatoric_confidence"]),
                    "combined_confidence_official": float(record["combined_confidence_official"]),
                    "predicted_magnitude_rms": magnitude,
                    "pearson_effect_accuracy": vector_pearson(pred_effect, truth_effect),
                    "frac_correct_direction_all": float(np.mean(pred_effect * truth_effect > 0)),
                    "frac_correct_direction_top20_de": float(
                        np.mean(pred_effect[top20] * truth_effect[top20] > 0)
                    ),
                    "rmse_effect_error": float(np.sqrt(np.mean((pred_effect - truth_effect) ** 2))),
                    "raw_pearson_effect_accuracy_sensitivity": vector_pearson(
                        pred_effect, raw_truth_effect
                    ),
                    "raw_frac_correct_direction_all_sensitivity": float(
                        np.mean(pred_effect * raw_truth_effect > 0)
                    ),
                    "raw_rmse_effect_error_sensitivity": float(
                        np.sqrt(np.mean((pred_effect - raw_truth_effect) ** 2))
                    ),
                    "true_magnitude_pca10_diagnostic_only": float(
                        np.sqrt(np.mean(truth_effect**2))
                    ),
                    "true_magnitude_raw_diagnostic_only": float(
                        np.sqrt(np.mean(raw_truth_effect**2))
                    ),
                }
            )
            count_audit.append(
                {"panel": panel, "task_id": task, "observed_cells": n_cells, "expected_cells": expected_cells}
            )
    tasks = pd.DataFrame(rows).sort_values(["panel", "locked_row_order"]).reset_index(drop=True)
    if len(tasks) != 48 or tasks.duplicated(["panel", "task_id"]).any():
        raise RuntimeError("E158 did not produce exactly 48 panel-task records")
    metric_columns = list(TARGETS) + ["combined_confidence_official", "predicted_magnitude_rms"]
    if not np.isfinite(tasks[metric_columns].to_numpy(float)).all():
        raise RuntimeError("E158 produced a non-finite score or endpoint")
    return tasks, {
        "n_test_cells": int(len(labels)),
        "n_tasks": int(len(tasks)),
        "normalization": "per-cell target_sum=10000 then log1p",
        "truth_primary": "PCA10-reconstructed task mean",
        "truth_sensitivity": "raw fixed-log-normalized task mean",
        "cell_count_audit_max_abs_delta": int(
            max(abs(item["observed_cells"] - item["expected_cells"]) for item in count_audit)
        ),
    }


def percentile(values: np.ndarray) -> tuple[float, float, int]:
    finite = np.asarray(values, float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < MIN_VALID_BOOT:
        raise RuntimeError(f"Only {len(finite)} valid bootstrap replicates; need {MIN_VALID_BOOT}")
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high), int(len(finite))


def target_bootstrap(
    tasks: pd.DataFrame,
    target: str,
    method: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = {
        panel: group.sort_values("locked_row_order").reset_index(drop=True)
        for panel, group in tasks.groupby("panel", sort=True)
    }
    panels = sorted(groups)
    rng = np.random.default_rng(fixed_seed(f"E158|{method}|{target}|paired-v1"))
    score_cols = list(SCORES.values())
    draws: dict[str, np.ndarray] = {}
    for panel in panels:
        for score in score_cols:
            draws[f"{panel}|{score}"] = np.full(N_BOOT, np.nan)
    for replicate in range(N_BOOT):
        for panel in panels:
            group = groups[panel]
            index = rng.integers(0, len(group), len(group))
            y = group[target].to_numpy(float)[index]
            for score in score_cols:
                x = group[score].to_numpy(float)[index]
                draws[f"{panel}|{score}"][replicate] = safe_correlation(x, y, method)

    association_rows = []
    delta_rows = []
    draw_frame: dict[str, object] = {"replicate": np.arange(N_BOOT, dtype=int)}
    sign = float(TARGETS[target]["favorable_sign"])
    for panel in panels:
        group = groups[panel]
        observed = {
            score: safe_correlation(
                group[score].to_numpy(float), group[target].to_numpy(float), method
            )
            for score in score_cols
        }
        for score_name, score in SCORES.items():
            low, high, valid = percentile(draws[f"{panel}|{score}"])
            association_rows.append(
                {
                    "scope": panel,
                    "method": method,
                    "score": score_name,
                    "target": target,
                    "target_role": TARGETS[target]["role"],
                    "expected_favorable_sign": int(sign),
                    "n_tasks": len(group),
                    "estimate": observed[score],
                    "favorable_estimate": sign * observed[score],
                    "bootstrap_ci95_low": low,
                    "bootstrap_ci95_high": high,
                    "bootstrap_valid": valid,
                }
            )
            draw_frame[f"{panel}__{score_name}"] = draws[f"{panel}|{score}"]
        delta_values = (
            draws[f"{panel}|{SCORES['combined_confidence']}"]
            - draws[f"{panel}|{SCORES['predicted_magnitude']}"]
        )
        low, high, valid = percentile(delta_values)
        observed_delta = observed[SCORES["combined_confidence"]] - observed[
            SCORES["predicted_magnitude"]
        ]
        delta_rows.append(
            {
                "scope": panel,
                "method": method,
                "score": "combined_confidence",
                "baseline": "predicted_magnitude",
                "target": target,
                "raw_delta": observed_delta,
                "favorable_delta": sign * observed_delta,
                "raw_bootstrap_ci95_low": low,
                "raw_bootstrap_ci95_high": high,
                "favorable_bootstrap_ci95_low": min(sign * low, sign * high),
                "favorable_bootstrap_ci95_high": max(sign * low, sign * high),
                "bootstrap_valid": valid,
            }
        )
        draw_frame[f"{panel}__delta_combined_minus_magnitude"] = delta_values

    for score_name, score in SCORES.items():
        panel_points = [
            safe_correlation(
                groups[panel][score].to_numpy(float),
                groups[panel][target].to_numpy(float),
                method,
            )
            for panel in panels
        ]
        macro_draw = np.mean(
            np.column_stack([draws[f"{panel}|{score}"] for panel in panels]), axis=1
        )
        low, high, valid = percentile(macro_draw)
        association_rows.append(
            {
                "scope": "two_panel_equal_macro",
                "method": method,
                "score": score_name,
                "target": target,
                "target_role": TARGETS[target]["role"],
                "expected_favorable_sign": int(sign),
                "n_tasks": len(tasks),
                "estimate": float(np.mean(panel_points)),
                "favorable_estimate": sign * float(np.mean(panel_points)),
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
                "bootstrap_valid": valid,
            }
        )
        draw_frame[f"two_panel_equal_macro__{score_name}"] = macro_draw
    macro_delta_draw = (
        np.asarray(draw_frame["two_panel_equal_macro__combined_confidence"])
        - np.asarray(draw_frame["two_panel_equal_macro__predicted_magnitude"])
    )
    low, high, valid = percentile(macro_delta_draw)
    macro_combined = next(
        row["estimate"]
        for row in association_rows
        if row["scope"] == "two_panel_equal_macro" and row["score"] == "combined_confidence"
    )
    macro_magnitude = next(
        row["estimate"]
        for row in association_rows
        if row["scope"] == "two_panel_equal_macro" and row["score"] == "predicted_magnitude"
    )
    delta_rows.append(
        {
            "scope": "two_panel_equal_macro",
            "method": method,
            "score": "combined_confidence",
            "baseline": "predicted_magnitude",
            "target": target,
            "raw_delta": float(macro_combined - macro_magnitude),
            "favorable_delta": sign * float(macro_combined - macro_magnitude),
            "raw_bootstrap_ci95_low": low,
            "raw_bootstrap_ci95_high": high,
            "favorable_bootstrap_ci95_low": min(sign * low, sign * high),
            "favorable_bootstrap_ci95_high": max(sign * low, sign * high),
            "bootstrap_valid": valid,
        }
    )
    draw_frame["two_panel_equal_macro__delta_combined_minus_magnitude"] = macro_delta_draw

    # P3/P4 share the same development data, seed, and training protocol and are
    # two disjoint SHA partitions of 48 tasks, not independent study replicates.
    # The frozen equal-panel macro remains primary; pooled-48 is a preregistered
    # dependence-aware sensitivity estimate and never changes the gate.
    pooled = tasks.sort_values(["panel", "locked_row_order"]).reset_index(drop=True)
    pooled_rng = np.random.default_rng(
        fixed_seed(f"E158|{method}|{target}|pooled-48-paired-v1")
    )
    pooled_draws = {score: np.full(N_BOOT, np.nan) for score in score_cols}
    for replicate in range(N_BOOT):
        index = pooled_rng.integers(0, len(pooled), len(pooled))
        y = pooled[target].to_numpy(float)[index]
        for score in score_cols:
            pooled_draws[score][replicate] = safe_correlation(
                pooled[score].to_numpy(float)[index], y, method
            )
    pooled_points = {
        score: safe_correlation(
            pooled[score].to_numpy(float), pooled[target].to_numpy(float), method
        )
        for score in score_cols
    }
    for score_name, score in SCORES.items():
        low, high, valid = percentile(pooled_draws[score])
        association_rows.append(
            {
                "scope": "pooled_48_task_sensitivity",
                "method": method,
                "score": score_name,
                "target": target,
                "target_role": TARGETS[target]["role"],
                "expected_favorable_sign": int(sign),
                "n_tasks": len(pooled),
                "estimate": pooled_points[score],
                "favorable_estimate": sign * pooled_points[score],
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
                "bootstrap_valid": valid,
            }
        )
        draw_frame[f"pooled_48_task_sensitivity__{score_name}"] = pooled_draws[score]
    pooled_delta_draw = (
        pooled_draws[SCORES["combined_confidence"]]
        - pooled_draws[SCORES["predicted_magnitude"]]
    )
    low, high, valid = percentile(pooled_delta_draw)
    pooled_delta = (
        pooled_points[SCORES["combined_confidence"]]
        - pooled_points[SCORES["predicted_magnitude"]]
    )
    delta_rows.append(
        {
            "scope": "pooled_48_task_sensitivity",
            "method": method,
            "score": "combined_confidence",
            "baseline": "predicted_magnitude",
            "target": target,
            "raw_delta": pooled_delta,
            "favorable_delta": sign * pooled_delta,
            "raw_bootstrap_ci95_low": low,
            "raw_bootstrap_ci95_high": high,
            "favorable_bootstrap_ci95_low": min(sign * low, sign * high),
            "favorable_bootstrap_ci95_high": max(sign * low, sign * high),
            "bootstrap_valid": valid,
        }
    )
    draw_frame["pooled_48_task_sensitivity__delta_combined_minus_magnitude"] = pooled_delta_draw
    return pd.DataFrame(association_rows), pd.DataFrame(delta_rows), pd.DataFrame(draw_frame)


def build_associations(tasks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    associations = []
    deltas = []
    primary_draws = None
    for target in TARGETS:
        association, delta, draws = target_bootstrap(tasks, target, "spearman")
        associations.append(association)
        deltas.append(delta)
        if target == "pearson_effect_accuracy":
            primary_draws = draws
    outer_association, outer_delta, _ = target_bootstrap(
        tasks, "pearson_effect_accuracy", "pearson"
    )
    associations.append(outer_association)
    deltas.append(outer_delta)
    if primary_draws is None:
        raise AssertionError("Primary bootstrap draws were not generated")
    return pd.concat(associations, ignore_index=True), pd.concat(deltas, ignore_index=True), primary_draws


def retained_mean(
    score: np.ndarray,
    target: np.ndarray,
    coverage: float,
    tie_order: np.ndarray | None = None,
) -> tuple[float, int]:
    n_keep = len(score) if coverage >= 1.0 else max(2, int(np.floor(len(score) * coverage)))
    score = np.asarray(score, float)
    if tie_order is None:
        tie_order = np.arange(len(score), dtype=int)
    order = np.lexsort((np.asarray(tie_order, int), -score))
    return float(np.mean(np.asarray(target, float)[order[:n_keep]])), int(n_keep)


def build_coverage(tasks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = {
        panel: group.sort_values("locked_row_order").reset_index(drop=True)
        for panel, group in tasks.groupby("panel", sort=True)
    }
    panels = sorted(groups)
    curve_rows = []
    delta_rows = []
    for coverage in COVERAGES:
        panel_points: dict[str, dict[str, float]] = {}
        for panel in panels:
            group = groups[panel]
            panel_points[panel] = {}
            for score_name, score in SCORES.items():
                value, n_keep = retained_mean(
                    group[score].to_numpy(float),
                    group["pearson_effect_accuracy"].to_numpy(float),
                    float(coverage),
                    group["locked_row_order"].to_numpy(int),
                )
                panel_points[panel][score_name] = value
                curve_rows.append(
                    {
                        "scope": panel,
                        "score": score_name,
                        "target": "pearson_effect_accuracy",
                        "coverage": float(coverage),
                        "n_retained": n_keep,
                        "retained_mean": value,
                        "focal_coverage": bool(float(coverage) in FOCAL_COVERAGES),
                    }
                )
        for score_name in SCORES:
            curve_rows.append(
                {
                    "scope": "two_panel_equal_macro",
                    "score": score_name,
                    "target": "pearson_effect_accuracy",
                    "coverage": float(coverage),
                    "n_retained": sum(
                        len(group)
                        if coverage >= 1.0
                        else max(2, int(np.floor(len(group) * coverage)))
                        for group in groups.values()
                    ),
                    "retained_mean": float(
                        np.mean([panel_points[panel][score_name] for panel in panels])
                    ),
                    "focal_coverage": bool(float(coverage) in FOCAL_COVERAGES),
                }
            )

        rng = np.random.default_rng(fixed_seed(f"E158|coverage|{coverage:.2f}|paired-v1"))
        panel_deltas = {panel: np.empty(N_BOOT, float) for panel in panels}
        for replicate in range(N_BOOT):
            for panel in panels:
                group = groups[panel]
                index = rng.integers(0, len(group), len(group))
                y = group["pearson_effect_accuracy"].to_numpy(float)[index]
                combined, _ = retained_mean(
                    group[SCORES["combined_confidence"]].to_numpy(float)[index],
                    y,
                    float(coverage),
                    group["locked_row_order"].to_numpy(int)[index],
                )
                magnitude, _ = retained_mean(
                    group[SCORES["predicted_magnitude"]].to_numpy(float)[index],
                    y,
                    float(coverage),
                    group["locked_row_order"].to_numpy(int)[index],
                )
                panel_deltas[panel][replicate] = combined - magnitude
        for panel in panels:
            low, high, valid = percentile(panel_deltas[panel])
            delta_rows.append(
                {
                    "scope": panel,
                    "coverage": float(coverage),
                    "target": "pearson_effect_accuracy",
                    "combined_minus_magnitude": panel_points[panel]["combined_confidence"]
                    - panel_points[panel]["predicted_magnitude"],
                    "bootstrap_ci95_low": low,
                    "bootstrap_ci95_high": high,
                    "bootstrap_valid": valid,
                    "focal_coverage": bool(float(coverage) in FOCAL_COVERAGES),
                }
            )
        macro_draw = np.mean(np.column_stack([panel_deltas[panel] for panel in panels]), axis=1)
        low, high, valid = percentile(macro_draw)
        macro_delta = float(
            np.mean(
                [
                    panel_points[panel]["combined_confidence"]
                    - panel_points[panel]["predicted_magnitude"]
                    for panel in panels
                ]
            )
        )
        delta_rows.append(
            {
                "scope": "two_panel_equal_macro",
                "coverage": float(coverage),
                "target": "pearson_effect_accuracy",
                "combined_minus_magnitude": macro_delta,
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
                "bootstrap_valid": valid,
                "focal_coverage": bool(float(coverage) in FOCAL_COVERAGES),
            }
        )
    return pd.DataFrame(curve_rows), pd.DataFrame(delta_rows)


def gate_results(associations: pd.DataFrame, deltas: pd.DataFrame) -> dict[str, object]:
    primary = associations[
        associations["method"].eq("spearman")
        & associations["target"].eq("pearson_effect_accuracy")
        & associations["score"].eq("combined_confidence")
    ].set_index("scope")
    primary_delta = deltas[
        deltas["method"].eq("spearman")
        & deltas["target"].eq("pearson_effect_accuracy")
        & deltas["scope"].eq("two_panel_equal_macro")
    ].iloc[0]
    direction = associations[
        associations["method"].eq("spearman")
        & associations["target"].eq("frac_correct_direction_all")
        & associations["score"].eq("combined_confidence")
    ].set_index("scope")
    signal = bool(
        primary.loc["Norman_P3", "estimate"] > 0
        and primary.loc["Norman_P4", "estimate"] > 0
        and primary.loc["two_panel_equal_macro", "bootstrap_ci95_low"] > 0
    )
    incremental = bool(signal and primary_delta["raw_bootstrap_ci95_low"] > 0)
    directional = bool(
        direction.loc["Norman_P3", "estimate"] > 0
        and direction.loc["Norman_P4", "estimate"] > 0
        and direction.loc["two_panel_equal_macro", "bootstrap_ci95_low"] > 0
    )
    return {
        "paper_endpoint_signal_confirmed": signal,
        "incremental_advantage_over_magnitude_confirmed": incremental,
        "directional_biology_confirmed": directional,
        "primary_panel_rho": {
            panel: float(primary.loc[panel, "estimate"])
            for panel in ["Norman_P3", "Norman_P4"]
        },
        "primary_macro_rho": float(primary.loc["two_panel_equal_macro", "estimate"]),
        "primary_macro_ci95": [
            float(primary.loc["two_panel_equal_macro", "bootstrap_ci95_low"]),
            float(primary.loc["two_panel_equal_macro", "bootstrap_ci95_high"]),
        ],
        "primary_macro_delta_vs_magnitude": float(primary_delta["raw_delta"]),
        "primary_macro_delta_ci95": [
            float(primary_delta["raw_bootstrap_ci95_low"]),
            float(primary_delta["raw_bootstrap_ci95_high"]),
        ],
    }


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(map(str, frame.columns))
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def make_figure(associations: pd.DataFrame, coverage: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    primary = associations[
        associations["method"].eq("spearman")
        & associations["target"].eq("pearson_effect_accuracy")
    ]
    scopes = [
        "Norman_P3",
        "Norman_P4",
        "two_panel_equal_macro",
        "pooled_48_task_sensitivity",
    ]
    labels = ["P3", "P4", "Equal macro", "Pooled 48"]
    colors = {"combined_confidence": "#3B6F8E", "predicted_magnitude": "#777777"}
    for offset, score in [(-0.12, "combined_confidence"), (0.12, "predicted_magnitude")]:
        subset = primary[primary["score"].eq(score)].set_index("scope").loc[scopes]
        x = np.arange(len(scopes)) + offset
        y = subset["estimate"].to_numpy(float)
        low = subset["bootstrap_ci95_low"].to_numpy(float)
        high = subset["bootstrap_ci95_high"].to_numpy(float)
        # Draw percentile interval endpoints directly because the interval
        # need not contain the plug-in estimate.
        axes[0].vlines(x, low, high, color=colors[score], linewidth=1.2)
        axes[0].hlines(low, x - 0.035, x + 0.035, color=colors[score], linewidth=1.2)
        axes[0].hlines(high, x - 0.035, x + 0.035, color=colors[score], linewidth=1.2)
        axes[0].scatter(
            x,
            y,
            s=24,
            color=colors[score],
            label=score.replace("_", " "),
            zorder=3,
        )
    axes[0].axhline(0, color="#B8B8B8", linewidth=0.8)
    axes[0].set_xticks(np.arange(len(scopes)), labels)
    axes[0].set_ylabel("Spearman rho with Pearson accuracy")
    axes[0].set_title("a  Locked confidence calibration")
    axes[0].legend(frameon=False, fontsize=8)

    macro = coverage[coverage["scope"].eq("two_panel_equal_macro")]
    for score in ["combined_confidence", "predicted_magnitude"]:
        subset = macro[macro["score"].eq(score)].sort_values("coverage")
        axes[1].plot(
            100 * subset["coverage"],
            subset["retained_mean"],
            marker="o",
            markersize=3,
            linewidth=1.5,
            color=colors[score],
            label=score.replace("_", " "),
        )
    axes[1].set_xlabel("Coverage retained (%)")
    axes[1].set_ylabel("Mean Pearson accuracy")
    axes[1].set_title("b  Selective prediction")
    axes[1].legend(frameon=False, fontsize=8)
    path = OUT / "figures/E158_PRIMARY_FORWARD_EVALUATION.svg"
    figure.savefig(path, facecolor="white")
    plt.close(figure)


def write_report(
    associations: pd.DataFrame,
    deltas: pd.DataFrame,
    coverage_deltas: pd.DataFrame,
    gates: dict[str, object],
) -> None:
    primary = associations[
        associations["method"].eq("spearman")
        & associations["target"].eq("pearson_effect_accuracy")
    ][["scope", "score", "estimate", "bootstrap_ci95_low", "bootstrap_ci95_high"]].round(4)
    delta = deltas[
        deltas["method"].eq("spearman")
        & deltas["target"].eq("pearson_effect_accuracy")
    ][["scope", "raw_delta", "raw_bootstrap_ci95_low", "raw_bootstrap_ci95_high"]].round(4)
    focal = coverage_deltas[
        coverage_deltas["scope"].eq("two_panel_equal_macro")
        & coverage_deltas["focal_coverage"]
    ][["coverage", "combined_minus_magnitude", "bootstrap_ci95_low", "bootstrap_ci95_high"]].round(4)
    report = f"""# E158｜PRESCRIBE P3/P4 严格前瞻评价报告

## 判定

- 论文口径信号确认：`{gates['paper_endpoint_signal_confirmed']}`
- 相对 predicted magnitude 的增量优势确认：`{gates['incremental_advantage_over_magnitude_confirmed']}`
- 方向生物学确认：`{gates['directional_biology_confirmed']}`

E158 在 P3、P4 的 E157 checkpoint 和 label-only 分数全部锁定且逐字节验证后，才一次性解封测试 X。两个面板各24 个、共48 个任务全部保留。

## 主终点

{markdown_table(primary)}

combined confidence 相对 predicted magnitude 的配对 rho 差：

{markdown_table(delta)}

## 预先指定的 coverage

{markdown_table(focal)}

## 文件路径

- 48 任务指标：`tables/E158_TASK_METRICS.csv`
- 全部关联与 bootstrap CI：`tables/E158_ASSOCIATIONS.csv`
- 配对 combined-vs-magnitude：`tables/E158_INCREMENTAL_VS_MAGNITUDE.csv`
- 主终点 10,000 次 draws：`tables/E158_PRIMARY_BOOTSTRAP_DRAWS.csv`
- coverage：`tables/E158_COVERAGE_CURVES.csv` 和 `tables/E158_COVERAGE_VS_MAGNITUDE.csv`
- 解封证据：`UNSEAL_EVENT.json`

## 边界

P3/P4 共用同一份 development 数据、种子和训练协议，只是 48 个 held-out tasks 的两个固定 SHA 分区，不是两项独立研究复现。`pooled_48_task_sensitivity` 已和冻结 panel-macro 并列报告，但不替换主 gate。PCA10-reconstructed truth 是预注册主终点，raw log-normalized truth 只是敏感性。top20 DE 方向准确度也是补充终点。主判定不会因这些结果更改。
"""
    (OUT / "reports/E158_REPORT.md").write_text(report, encoding="utf-8")


def output_manifest() -> pd.DataFrame:
    candidates = [
        path
        for folder in [OUT / "tables", OUT / "figures", OUT / "reports"]
        for path in folder.rglob("*")
        if path.is_file()
    ]
    candidates.extend(
        path
        for path in [
            OUT / "README_先看这个.md",
            OUT / "UNSEAL_EVENT.json",
            OUT / "INPUT_MANIFEST_PRE_UNSEAL.csv",
        ]
        if path.is_file()
    )
    return pd.DataFrame(
        [
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(candidates)
        ]
    )


def select_append_only_attempt() -> Path:
    """Choose attempt_001 or the next attempt only after a preserved failure."""
    BASE_OUT.mkdir(parents=True, exist_ok=True)
    if (BASE_OUT / "RUN_STATUS.json").exists() or (BASE_OUT / "UNSEAL_EVENT.json").exists():
        raise RuntimeError("Legacy E158 output exists in the contract directory; manual audit required")
    attempts = []
    for path in BASE_OUT.glob("attempt_[0-9][0-9][0-9]"):
        if path.is_dir():
            attempts.append((int(path.name.split("_")[1]), path))
    attempts.sort()
    if not attempts:
        candidate = BASE_OUT / "attempt_001"
    else:
        expected = list(range(1, attempts[-1][0] + 1))
        observed = [index for index, _ in attempts]
        if observed != expected:
            raise RuntimeError(f"E158 attempt numbering has a gap: {observed}")
        last_index, last_path = attempts[-1]
        status_path = last_path / "RUN_STATUS.json"
        if not status_path.is_file():
            raise RuntimeError(f"Latest E158 attempt has no preserved status: {last_path}")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        phase = str(status.get("phase", ""))
        if phase == "complete_strict_forward_evaluation_no_posthoc_changes":
            raise RuntimeError("E158 strict forward evaluation is already complete")
        if not phase.startswith("failed_"):
            raise RuntimeError(
                f"Latest E158 attempt is not terminal and cannot be overwritten: {phase}"
            )
        candidate = BASE_OUT / f"attempt_{last_index + 1:03d}"
    # Atomic claim: exactly one concurrent process can own this append-only attempt.
    candidate.mkdir(exist_ok=False)
    return candidate


def main() -> None:
    global OUT
    OUT = select_append_only_attempt()
    status_path = OUT / "RUN_STATUS.json"
    unseal_path = OUT / "UNSEAL_EVENT.json"
    if status_path.exists() or unseal_path.exists():
        raise FileExistsError(
            "E158 has an existing STATUS/unseal record; preserve it and use a new attempt directory"
        )
    started = time.time()
    test_x_unsealed = False
    for folder in [OUT / "tables", OUT / "figures", OUT / "reports"]:
        folder.mkdir(exist_ok=True)
    update_status(
        status_path,
        experiment="E158_prescribe_norman_p3p4_forward_evaluation",
        phase="started_lock_verification_no_raw_access",
        started_at=now(),
        n_bootstrap=N_BOOT,
        coverage_grid=COVERAGES.tolist(),
        test_data_unsealed=False,
        raw_file_bytes_accessed=False,
        test_X_rows_materialized=False,
    )
    try:
        locked_tables, lock_audit, input_rows, condition_audit = verify_all_locked_inputs()
        transform = load_frozen_transform()
        score_semantics_audit = verify_locked_score_semantics(locked_tables, transform)
        input_rows = pd.concat(
            [
                input_rows,
                pd.DataFrame(
                    [
                        {
                            "role": "E156_train_only_PCA_model",
                            "path": str(transform["model_path"]),
                            "sha256": str(transform["model_sha256"]),
                        },
                        {
                            "role": "future_raw_Norman_source_expected_not_accessed_pre_unseal",
                            "path": str(RAW),
                            "sha256": EXPECTED_RAW_SHA256,
                        },
                    ]
                ),
            ],
            ignore_index=True,
        )
        input_manifest_path = OUT / "INPUT_MANIFEST_PRE_UNSEAL.csv"
        input_manifest_hash = write_once(
            input_manifest_path,
            input_rows.sort_values(["role", "path"]).to_csv(index=False).encode("utf-8"),
        )
        update_status(
            status_path,
            phase="both_E157_runs_and_all_locked_assets_verified_no_raw_access",
            both_E157_complete_and_hash_verified=True,
            lock_verification_completed_at=now(),
            lock_audit=lock_audit,
            input_manifest_sha256=input_manifest_hash,
            pca_model_sha256=transform["model_sha256"],
            gene_order_sha256=transform["gene_order_sha256"],
            control_sha256=transform["control_sha256"],
            locked_score_semantics_audit=score_semantics_audit,
            test_data_unsealed=False,
            raw_file_bytes_accessed=False,
            test_X_rows_materialized=False,
        )

        unseal = {
            "event": "E158_one_time_test_expression_unseal_after_both_E157_locks",
            "unsealed_at": now(),
            "raw_path": str(RAW),
            "expected_raw_sha256": EXPECTED_RAW_SHA256,
            "input_manifest_sha256": input_manifest_hash,
            "runner_sha256": lock_audit["runner_sha256"],
            "contract_sha256": lock_audit["contract_sha256"],
            "git_head_at_unseal": lock_audit["git_head_at_unseal"],
            "panels": lock_audit["panels"],
            "locked_score_semantics_audit": score_semantics_audit,
            "irreversible_record": True,
        }
        unseal_event_hash = write_once(
            unseal_path,
            (json.dumps(unseal, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        )
        test_x_unsealed = True
        update_status(
            status_path,
            phase="test_data_unsealed_raw_hash_verification_started",
            test_data_unsealed=True,
            unseal_event_sha256=unseal_event_hash,
            raw_file_bytes_accessed=True,
            test_X_rows_materialized=False,
        )
        raw_hash = require_file_hash(
            RAW, EXPECTED_RAW_SHA256, "frozen raw Norman H5AD", allow_raw=True
        )
        update_status(
            status_path,
            phase="raw_hash_verified_test_X_access_authorized",
            raw_sha256=raw_hash,
            raw_hash_verified_at=now(),
        )

        tasks, truth_audit = unseal_and_compute_tasks(
            locked_tables, transform, condition_audit, status_path
        )
        tasks.to_csv(OUT / "tables/E158_TASK_METRICS.csv", index=False)
        associations, deltas, primary_draws = build_associations(tasks)
        coverage, coverage_deltas = build_coverage(tasks)
        associations.to_csv(OUT / "tables/E158_ASSOCIATIONS.csv", index=False)
        deltas.to_csv(OUT / "tables/E158_INCREMENTAL_VS_MAGNITUDE.csv", index=False)
        primary_draws.to_csv(OUT / "tables/E158_PRIMARY_BOOTSTRAP_DRAWS.csv", index=False)
        coverage.to_csv(OUT / "tables/E158_COVERAGE_CURVES.csv", index=False)
        coverage_deltas.to_csv(OUT / "tables/E158_COVERAGE_VS_MAGNITUDE.csv", index=False)
        gates = gate_results(associations, deltas)
        make_figure(associations, coverage)
        write_report(associations, deltas, coverage_deltas, gates)
        (OUT / "README_先看这个.md").write_text(
            "# E158 先看这个\n\n"
            "先读 `../ANALYSIS_CONTRACT.md`，再读 `reports/E158_REPORT.md`。\n"
            "`UNSEAL_EVENT.json` 记录两个 E157 锁定后的首次测试表达解封。\n",
            encoding="utf-8",
        )
        manifest = output_manifest()
        manifest.to_csv(OUT / "OUTPUT_MANIFEST.csv", index=False)
        update_status(
            status_path,
            phase="complete_strict_forward_evaluation_no_posthoc_changes",
            finished_at=now(),
            runtime_seconds=round(time.time() - started, 3),
            test_data_unsealed=True,
            raw_file_bytes_accessed=True,
            test_X_rows_materialized=True,
            test_endpoint_computed=True,
            truth_audit=truth_audit,
            gates=gates,
            output_manifest_sha256=sha256_file(OUT / "OUTPUT_MANIFEST.csv"),
            output_manifest_rows=int(len(manifest)),
            task_table_sha256=sha256_file(OUT / "tables/E158_TASK_METRICS.csv"),
            primary_bootstrap_draws_sha256=sha256_file(
                OUT / "tables/E158_PRIMARY_BOOTSTRAP_DRAWS.csv"
            ),
            analysis_contract_sha256=sha256_file(CONTRACT),
            runner_sha256=sha256_file(Path(__file__).resolve()),
            target_truth_used_to_change_task_set_or_score=False,
            post_unblinding_metric_or_gate_change=False,
        )
        print(status_path.read_text(encoding="utf-8"))
    except Exception as exc:
        irreversible_unseal = bool(test_x_unsealed or unseal_path.is_file())
        update_status(
            status_path,
            phase=(
                "failed_after_irreversible_test_unseal_preserve_attempt"
                if irreversible_unseal
                else "failed_preflight_before_any_raw_access"
            ),
            failed_at=now(),
            runtime_seconds=round(time.time() - started, 3),
            test_data_unsealed=irreversible_unseal,
            error=repr(exc),
            traceback=traceback.format_exc(),
        )
        raise


if __name__ == "__main__":
    main()
