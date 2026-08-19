#!/usr/bin/env python3
"""E162: train native PRESCRIBE on frozen Wessels development assets.

The formal run is deliberately split from E163 evaluation.  This runner never
accepts a source-dataset path, never constructs a test graph/DataLoader, and
never reads test expression or truth.  Test conditions are queried only as
strings after the main-seed validation non-degeneracy gate has passed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PRESCRIBE = Path("/home/yyf/archive/external/PRESCRIBE")
DATA_ROOT = Path("/home/yyf/data/safeconf_e161_prescribe/wessels_e160")
PRESCRIBE_DATA_LINK = PRESCRIBE / "data/wessels_e160"
E160 = ROOT / "docs/实验结果/E160_wessels_combination_contract_20260714"
E161 = ROOT / "docs/实验结果/E161_wessels_trainval_preprocess_20260714"
BASE_OUT = ROOT / "docs/实验结果/E162_wessels_prescribe_native_20260714"
CONTRACT = BASE_OUT / "ANALYSIS_CONTRACT.md"
INTERFACE = DATA_ROOT / "E161_E162_INTERFACE.json"
E160_STATUS = E160 / "freeze/RUN_STATUS.json"
E160_SPLIT = E160 / "freeze/manifests/E160_set2conditions.json"
E161_STATUS = E161 / "release/RUN_STATUS.json"
E161_ASSET_MANIFEST = E161 / "release/tables/E161_ASSET_MANIFEST.csv"
LOCKED_MODEL_ROOT = Path("/home/yyf/data/safeconf_e162_locked_models")

EXPECTED_PYTHON = Path("/home/yyf/.conda/envs/prescribe_env/bin/python")
EXPECTED_PRESCRIBE_COMMIT = "6f7264a205aaff654a9594863c5c10b656f88ebe"
EXPECTED_INTERFACE_SCHEMA = "safeconf_e161_to_e162_v2"
SEEDS = (3407, 3408, 3409)
MAIN_SEED = 3407
SEED_ROLE = {3407: "main", 3408: "training_sensitivity", 3409: "training_sensitivity"}
BATCH_SIZE = 512
ACCUMULATE_GRAD_BATCHES = 1
N_WARMUP_EPOCHS = 5
N_EPOCHS = 50
N_PCA = 10
N_EQUIVALENCE_TASKS = 8
GRAPH_X_ATOL = 1e-6
FORWARD_EQUIVALENCE_ATOL = 1e-5
RAW_LOG_STD_THRESHOLD = 1e-6
BASELINE_STD_THRESHOLD = 1e-12

EXPECTED_COUNTS = {
    "train_conditions": 72,
    "val_conditions": 24,
    "test_conditions": 48,
    "train_graphs": 11_779,
    "val_graphs": 5_102,
    "test_graphs": 0,
    "selected_genes": 2_023,
}
EXPECTED_VERSIONS = {
    "anndata": "0.10.8",
    "scanpy": "1.10.3",
    "numpy": "1.26.4",
    "pandas": "2.3.3",
    "scipy": "1.13.1",
    "scikit-learn": "1.6.1",
    "scikit-misc": "0.3.1",
    "h5py": "3.14.0",
    "lightning": "2.6.0",
    "torch": "2.1.2",
    "torch-geometric": "2.6.1",
}
LOCKED_WORKTREE_SOURCE_SHA256 = {
    "gears/__init__.py": "3cdc747e61b16e073873d7f5ccb4f7c872d921c5355f2615049de09f279233ee",
    "gears/pertdata.py": "d7316bc19fc70d78c78d0dabf126df161ae861a6620d85a0d16aeaeee27ba59c",
    "gears/utils.py": "89d1da79df60d14d929aed05ec904b4ef2664855abe89b5bce5112b88f80395a",
    "gears/data_utils.py": "7043e80d4280cd81ec2ff6c78609235f407a4cd3dc3f337f3e53804e16c537fc",
    "src/data/pertdata.py": "f5247ceb8cd5e8a5782c74d1d4e17350dfbed2e2c911f56fb4bf9f69344acc77",
    "src/data/dataloader.py": "e8ac66674935ecd32d0d029169160f97abdfe6361c36ce9ff089598390e30362",
}
LOCKED_EXTERNAL_ASSET_SHA256 = {
    "data/gene2go_all.pkl": "f145c5e84a53048d87942a417d870a4f2d8db50200b96e492b358c13aba8c771",
    "scLLM_weights/scGPT/embedding.pkl": "9a5be69676bc09fbf996ae7be1d4faa09c9f32abbf733f33fc130153829ad8ce",
}
TRACKED_SOURCE_PREFIXES = ("gears/", "src/")
TRACKED_SOURCE_SINGLETONS = ("Step2_train.py",)
TERMINAL_PREFIXES = ("complete_", "failed_")
FROZEN_MAIN_FAILURE_PHASES = {
    "failed_main_validation_nondegeneracy_gate_no_test_label_query",
    "failed_main_test_nondegeneracy_gate_no_E163_unseal",
}
LISTMLE_LIMITATION = (
    "The current native ListMLE path is permutation-insensitive for the supplied "
    "E-distance vector; y_n is retained for interface compatibility and is not "
    "claimed as effective ranking supervision."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("preflight", "formal"))
    parser.add_argument("--gpu-index", type=int, default=0)
    return parser.parse_args()


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_bytes(path: Path, payload: bytes, *, replace: bool = True) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256_bytes(payload)
    if path.exists() and not replace:
        if not path.is_file() or path.is_symlink() or sha256_file(path) != digest:
            raise RuntimeError(f"Immutable artifact differs: {path}")
        return digest
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    if not replace and path.exists():
        temporary.unlink()
        raise FileExistsError(path)
    temporary.replace(path)
    fsync_directory(path.parent)
    return digest


def atomic_json(path: Path, payload: dict[str, Any], *, replace: bool = True) -> str:
    return atomic_bytes(path, canonical_json_bytes(payload), replace=replace)


def atomic_csv(path: Path, frame: pd.DataFrame, *, replace: bool = True) -> str:
    return atomic_bytes(path, frame.to_csv(index=False).encode("utf-8"), replace=replace)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def require_data_relative_path(value: str | Path, role: str) -> Path:
    """Resolve one manifest path while rejecting absolute/parent/symlink escapes."""
    relative = Path(str(value))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise RuntimeError(f"{role} is not a safe relative E161 asset path: {value}")
    candidate = DATA_ROOT / relative
    resolved = candidate.resolve()
    try:
        resolved.relative_to(DATA_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError(f"{role} escapes the E161 data root: {value}") from exc
    if candidate.is_symlink():
        raise RuntimeError(f"{role} is an asset-level symlink: {candidate}")
    return candidate


def update_status(path: Path, **updates: Any) -> dict[str, Any]:
    current = load_json(path) if path.exists() else {}
    current.update(updates)
    atomic_json(path, current)
    return current


def package_versions() -> dict[str, str]:
    observed: dict[str, str] = {}
    for distribution, expected in EXPECTED_VERSIONS.items():
        actual = importlib.metadata.version(distribution)
        if distribution == "torch":
            actual = actual.split("+")[0]
        if actual != expected:
            raise RuntimeError(f"Runtime version mismatch for {distribution}: {actual} != {expected}")
        observed[distribution] = importlib.metadata.version(distribution)
    return observed


def python_gate() -> dict[str, Any]:
    executable = Path(sys.executable).resolve()
    if executable != EXPECTED_PYTHON.resolve():
        raise RuntimeError(f"E162 requires {EXPECTED_PYTHON}, found {executable}")
    if sys.version_info[:3] != (3, 9, 25):
        raise RuntimeError(f"E162 requires Python 3.9.25, found {sys.version.split()[0]}")
    return {
        "executable": str(executable),
        "python": sys.version.split()[0],
        "packages": package_versions(),
    }


def git_blob_gate(*, require_committed_runner: bool) -> dict[str, Any]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    files = {
        "runner": Path(__file__).resolve(),
        "contract": CONTRACT,
        "e160_contract": E160 / "ANALYSIS_CONTRACT.md",
        "e160_status": E160_STATUS,
        "e161_contract": E161 / "ANALYSIS_CONTRACT.md",
        "e161_status": E161_STATUS,
        "e161_asset_manifest": E161_ASSET_MANIFEST,
    }
    rows: list[dict[str, Any]] = []
    for role, path in files.items():
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        relative = path.relative_to(ROOT).as_posix()
        try:
            committed = subprocess.check_output(
                ["git", "show", f"HEAD:{relative}"],
                cwd=ROOT,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            if role in {"runner", "contract"} and not require_committed_runner:
                rows.append({
                    "role": role,
                    "path": relative,
                    "working_sha256": sha256_file(path),
                    "committed": False,
                    "matches_head_blob": False,
                })
                continue
            raise RuntimeError(f"Required E162 input is not committed at HEAD: {relative}")
        working = sha256_file(path)
        committed_hash = sha256_bytes(committed)
        if working != committed_hash:
            if role in {"runner", "contract"} and not require_committed_runner:
                rows.append({
                    "role": role,
                    "path": relative,
                    "working_sha256": working,
                    "committed_sha256": committed_hash,
                    "committed": True,
                    "matches_head_blob": False,
                })
                continue
            raise RuntimeError(f"Working file differs from HEAD blob: {relative}")
        rows.append({
            "role": role,
            "path": relative,
            "working_sha256": working,
            "committed_sha256": committed_hash,
            "committed": True,
            "matches_head_blob": True,
        })
    return {"git_head": head, "files": rows}


def upstream_source_gate(*, hash_payloads: bool) -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "-C", str(PRESCRIBE), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != EXPECTED_PRESCRIBE_COMMIT:
        raise RuntimeError(f"PRESCRIBE commit mismatch: {commit}")
    tracked = subprocess.check_output(
        ["git", "-C", str(PRESCRIBE), "ls-tree", "-r", "--name-only", commit], text=True
    ).splitlines()
    relevant = sorted(
        path for path in tracked
        if path in TRACKED_SOURCE_SINGLETONS
        or (path.endswith(".py") and path.startswith(TRACKED_SOURCE_PREFIXES))
    )
    rows: list[dict[str, Any]] = []
    for relative in relevant:
        path = PRESCRIBE / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        if relative in LOCKED_WORKTREE_SOURCE_SHA256:
            expected = LOCKED_WORKTREE_SOURCE_SHA256[relative]
        else:
            committed = subprocess.check_output(
                ["git", "-C", str(PRESCRIBE), "show", f"{commit}:{relative}"]
            )
            expected = sha256_bytes(committed)
        actual = sha256_file(path) if hash_payloads else None
        if actual is not None and actual != expected:
            raise RuntimeError(f"PRESCRIBE source bytes changed: {relative}")
        rows.append({
            "relative_path": relative,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "bytes": path.stat().st_size,
        })
    for relative, expected in LOCKED_EXTERNAL_ASSET_SHA256.items():
        path = PRESCRIBE / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        actual = sha256_file(path) if hash_payloads else None
        if actual is not None and actual != expected:
            raise RuntimeError(f"PRESCRIBE frozen external asset changed: {relative}")
        rows.append({
            "relative_path": relative,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "bytes": path.stat().st_size,
        })
    return {"commit": commit, "files": rows}


def verify_interface_metadata(*, hash_large_assets: bool) -> dict[str, Any]:
    if not DATA_ROOT.is_dir() or DATA_ROOT.is_symlink():
        raise RuntimeError(f"E161 data root is not a regular directory: {DATA_ROOT}")
    if not PRESCRIBE_DATA_LINK.is_symlink():
        raise RuntimeError(f"Required PRESCRIBE data link is missing: {PRESCRIBE_DATA_LINK}")
    if PRESCRIBE_DATA_LINK.resolve() != DATA_ROOT.resolve():
        raise RuntimeError("PRESCRIBE wessels_e160 link points to a different release")
    interface = load_json(INTERFACE)
    if interface.get("schema") != EXPECTED_INTERFACE_SCHEMA:
        raise RuntimeError("E162 rejects any E161 interface other than v2")
    if Path(str(interface.get("data_root"))).resolve() != DATA_ROOT.resolve():
        raise RuntimeError("E161 interface data_root changed")
    if interface.get("data_name") != "wessels_e160" or interface.get("seed") != MAIN_SEED:
        raise RuntimeError("E161 interface identity mismatch")
    if interface.get("split_conditions") != {"train": 72, "val": 24, "test": 48}:
        raise RuntimeError("E161 split counts changed")
    if interface.get("development_graphs") != {"train": 11779, "val": 5102, "test": 0}:
        raise RuntimeError("E161 development graph counts changed")
    if interface.get("n_selected_genes") != EXPECTED_COUNTS["selected_genes"]:
        raise RuntimeError("E161 selected-gene count changed")
    false_flags = (
        "test_X_rows_indexed_materialized_or_transformed",
        "engineered_construct_X_columns_indexed_or_materialized",
        "guide_barcode_X_columns_indexed_or_materialized",
        "excluded_X_columns_indexed_or_materialized",
    )
    if any(interface.get(key) is not False for key in false_flags):
        raise RuntimeError("E161 interface does not certify its sealed-expression boundary")
    e161_status = load_json(E161_STATUS)
    if e161_status.get("phase") != "complete_preprocessing_and_dev_graphs_no_training_no_test_X_access":
        raise RuntimeError("E161 release is not preprocessing-complete")
    if any(e161_status.get(key) is not False for key in (
        "test_X_rows_indexed", "test_X_rows_materialized", "test_X_rows_transformed",
        "model_training_started", "predictions_generated", "test_endpoint_computed",
    )):
        raise RuntimeError("E161 release status violates E162 input boundary")
    published = pd.read_csv(E161_ASSET_MANIFEST)
    if set(published.columns) != {"relative_path", "bytes", "sha256"}:
        raise RuntimeError("E161 asset manifest schema changed")
    if published["relative_path"].astype(str).duplicated().any():
        raise RuntimeError("E161 asset manifest contains duplicate relative paths")
    manifest_records = {str(row.relative_path): row for row in published.itertuples(index=False)}
    interface_assets = {str(k): str(v) for k, v in interface["asset_sha256"].items()}
    if set(manifest_records) != set(interface_assets) | {"E161_E162_INTERFACE.json"}:
        raise RuntimeError("E161 interface/asset-manifest allowlists differ")
    if sha256_file(INTERFACE) != str(manifest_records["E161_E162_INTERFACE.json"].sha256):
        raise RuntimeError("E161 interface hash differs from release manifest")
    rows: list[dict[str, Any]] = []
    for relative, expected in sorted(interface_assets.items()):
        path = require_data_relative_path(relative, f"E161 manifest asset {relative}")
        record = manifest_records[relative]
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(record.bytes) or str(record.sha256) != expected:
            raise RuntimeError(f"E161 manifest metadata changed: {relative}")
        large = relative in {"perturb_processed.h5ad", "data_pyg/cell_graphs.pkl"}
        actual = sha256_file(path) if hash_large_assets or not large else None
        if actual is not None and actual != expected:
            raise RuntimeError(f"E161 asset changed: {relative}")
        rows.append({
            "relative_path": relative,
            "bytes": path.stat().st_size,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "payload_hashed": actual is not None,
        })
    return {
        "interface": interface,
        "interface_sha256": sha256_file(INTERFACE),
        "release_status_sha256": sha256_file(E161_STATUS),
        "release_manifest_sha256": sha256_file(E161_ASSET_MANIFEST),
        "asset_rows": rows,
        "data_link": str(PRESCRIBE_DATA_LINK),
        "data_link_target": str(PRESCRIBE_DATA_LINK.resolve()),
    }


def verify_e160_metadata() -> dict[str, Any]:
    status = load_json(E160_STATUS)
    if status.get("phase") != "requirements_frozen_test_expression_unopened":
        raise RuntimeError("E160 freeze status changed")
    if status.get("raw_X_values_indexed_or_materialized") is not False:
        raise RuntimeError("E160 no longer certifies unopened expression")
    if status.get("n_total_train_conditions") != 72 or status.get("n_pair_val") != 24 or status.get("n_pair_test") != 48:
        raise RuntimeError("E160 frozen split sizes changed")
    split = load_json(E160_SPLIT)
    if set(split) != {"train", "val", "test"}:
        raise RuntimeError("E160 split keys changed")
    if {role: len(split[role]) for role in split} != {"train": 72, "val": 24, "test": 48}:
        raise RuntimeError("E160 split JSON sizes changed")
    all_conditions = [str(value) for role in ("train", "val", "test") for value in split[role]]
    if len(all_conditions) != len(set(all_conditions)):
        raise RuntimeError("E160 split contains overlapping conditions")
    if sha256_file(E160_SPLIT) != status["artifact_sha256"]["freeze/manifests/E160_set2conditions.json"]:
        raise RuntimeError("E160 split JSON hash changed")
    return {
        "status_sha256": sha256_file(E160_STATUS),
        "split_sha256": sha256_file(E160_SPLIT),
        "split": split,
    }


def metadata_preflight(*, formal: bool) -> dict[str, Any]:
    runtime = python_gate()
    git_gate = git_blob_gate(require_committed_runner=formal)
    e160 = verify_e160_metadata()
    e161 = verify_interface_metadata(hash_large_assets=formal)
    sources = upstream_source_gate(hash_payloads=True)
    interface_split_path = require_data_relative_path(
        e161["interface"]["paths"]["split_pickle"], "E161 split pickle"
    )
    if sha256_file(interface_split_path) != e161["interface"]["asset_sha256"][interface_split_path.name]:
        raise RuntimeError("E161 split pickle changed")
    with interface_split_path.open("rb") as handle:
        interface_split = pickle.load(handle)
    if not isinstance(interface_split, dict) or set(interface_split) != {"train", "val", "test"}:
        raise RuntimeError("E161 split pickle schema changed")
    normalized_interface_split = {
        role: [str(value) for value in interface_split[role]]
        for role in ("train", "val", "test")
    }
    if normalized_interface_split != e160["split"]:
        raise RuntimeError("E161 split pickle differs from the ordered E160 JSON")
    payload = {
        "runtime": runtime,
        "git": git_gate,
        "e160_status_sha256": e160["status_sha256"],
        "e160_split_sha256": e160["split_sha256"],
        "e161_interface_sha256": e161["interface_sha256"],
        "e161_release_status_sha256": e161["release_status_sha256"],
        "e161_release_manifest_sha256": e161["release_manifest_sha256"],
        "e161_assets": e161["asset_rows"],
        "prescribe_commit": sources["commit"],
        "prescribe_sources": sources["files"],
        "data_link": e161["data_link"],
        "data_link_target": e161["data_link_target"],
        "split": e160["split"],
        "interface": e161["interface"],
        "development_h5ad_opened": False,
        "development_graph_cache_opened": False,
        "test_X_accessed": False,
        "test_truth_accessed": False,
    }
    fingerprint_view = dict(payload)
    fingerprint_view["git"] = {
        "git_head": payload["git"]["git_head"],
        "files": [
            {
                "role": row["role"],
                "path": row["path"],
                "working_sha256": row["working_sha256"],
            }
            for row in payload["git"]["files"]
        ]
    }
    payload["gate_fingerprint_sha256"] = sha256_bytes(canonical_json_bytes(fingerprint_view))
    return payload


def select_append_only_attempt(gate_fingerprint: str) -> tuple[Path, bool]:
    BASE_OUT.mkdir(parents=True, exist_ok=True)
    attempts: list[tuple[int, Path]] = []
    for path in BASE_OUT.glob("attempt_[0-9][0-9][0-9]"):
        if path.is_dir() and not path.is_symlink():
            attempts.append((int(path.name.split("_")[1]), path))
    attempts.sort()
    if attempts:
        observed = [value for value, _ in attempts]
        if observed != list(range(1, observed[-1] + 1)):
            raise RuntimeError(f"E162 append-only attempt numbering has a gap: {observed}")
        index, latest = attempts[-1]
        status_path = latest / "RUN_STATUS.json"
        if not status_path.is_file():
            raise RuntimeError(f"Latest E162 attempt has no status: {latest}")
        latest_status = load_json(status_path)
        phase = str(latest_status.get("phase", ""))
        if not phase.startswith(TERMINAL_PREFIXES):
            if latest_status.get("gate_fingerprint_sha256") == gate_fingerprint:
                return latest, False
            update_status(
                status_path,
                phase="failed_interrupted_fingerprint_changed_no_resume",
                failed_at=now(),
                replacement_gate_fingerprint_sha256=gate_fingerprint,
                interrupted_resume_bitwise_identity_claimed=False,
            )
        if (
            phase in FROZEN_MAIN_FAILURE_PHASES
            and latest_status.get("gate_fingerprint_sha256") == gate_fingerprint
        ):
            raise RuntimeError(
                "The frozen main validation/test gate already failed under this exact "
                "E162 fingerprint; an identical repeat is forbidden"
            )
        if (
            phase.startswith("complete_")
            and latest_status.get("gate_fingerprint_sha256") == gate_fingerprint
        ):
            raise RuntimeError("E162 already completed under this exact frozen fingerprint")
        candidate = BASE_OUT / f"attempt_{index + 1:03d}"
    else:
        candidate = BASE_OUT / "attempt_001"
    candidate.mkdir(mode=0o755)
    return candidate, True


def preflight_output(payload: dict[str, Any]) -> dict[str, Any]:
    git_runner = next(row for row in payload["git"]["files"] if row["role"] == "runner")
    git_contract = next(
        row for row in payload["git"]["files"] if row["role"] == "contract"
    )
    return {
        "experiment": "E162_wessels_prescribe_native",
        "mode": "preflight",
        "phase": "metadata_hash_contracts_passed_no_development_payload_opened",
        "runner_committed_and_matching": bool(git_runner["matches_head_blob"]),
        "contract_committed_and_matching": bool(git_contract["matches_head_blob"]),
        "git_head": payload["git"]["git_head"],
        "gate_fingerprint_sha256": payload["gate_fingerprint_sha256"],
        "interface_schema": payload["interface"]["schema"],
        "split_conditions": payload["interface"]["split_conditions"],
        "development_graphs": payload["interface"]["development_graphs"],
        "development_h5ad_opened": False,
        "development_graph_cache_opened": False,
        "test_X_accessed": False,
        "test_truth_accessed": False,
    }


def import_native_stack():
    """Import PRESCRIBE only inside formal execution and verify import provenance."""
    old_cwd = Path.cwd()
    old_path = list(sys.path)
    os.chdir(PRESCRIBE)
    sys.path.insert(0, str(PRESCRIBE))
    try:
        from gears import PertData  # noqa: PLC0415
        from src.data.pertdata import Get_Graph  # noqa: PLC0415
        from Step2_train import build_model  # noqa: PLC0415
        from src.model import (  # noqa: PLC0415
            NaturalPosteriorNetworkFlowLightningModule,
            NaturalPosteriorNetworkLightningModule,
        )

        imported = {
            "PertData": Path(inspect.getfile(PertData)).resolve(),
            "Get_Graph": Path(inspect.getfile(Get_Graph)).resolve(),
            "build_model": Path(inspect.getfile(build_model)).resolve(),
            "main_module": Path(inspect.getfile(NaturalPosteriorNetworkLightningModule)).resolve(),
        }
        expected = {
            "PertData": (PRESCRIBE / "gears/pertdata.py").resolve(),
            "Get_Graph": (PRESCRIBE / "src/data/pertdata.py").resolve(),
            "build_model": (PRESCRIBE / "Step2_train.py").resolve(),
            "main_module": (PRESCRIBE / "src/model/lightening_module.py").resolve(),
        }
        if imported != expected:
            raise RuntimeError(f"Imported PRESCRIBE provenance mismatch: {imported}")
        return SimpleNamespace(
            PertData=PertData,
            Get_Graph=Get_Graph,
            build_model=build_model,
            WarmupModule=NaturalPosteriorNetworkFlowLightningModule,
            MainModule=NaturalPosteriorNetworkLightningModule,
            old_cwd=old_cwd,
            old_path=old_path,
        )
    except Exception:
        sys.path[:] = old_path
        os.chdir(old_cwd)
        raise


def restore_import_context(native: SimpleNamespace) -> None:
    sys.path[:] = native.old_path
    os.chdir(native.old_cwd)


def load_development_pertdata(preflight: dict[str, Any]):
    """Direct E161 adapter; never call PRESCRIBE LoadData."""
    native = import_native_stack()
    interface = preflight["interface"]
    split = preflight["split"]
    gene_set_path = require_data_relative_path(
        f"frozen_pert_gene_set_{MAIN_SEED}.pkl", "E161 frozen perturbation gene set"
    )
    with gene_set_path.open("rb") as handle:
        frozen_genes = pickle.load(handle)
    expected_genes = sorted({
        gene
        for role in ("train", "val", "test")
        for condition in split[role]
        for gene in str(condition).split("+")
        if gene != "ctrl"
    })
    if sorted(map(str, frozen_genes)) != expected_genes or len(expected_genes) != 27:
        restore_import_context(native)
        raise RuntimeError("Frozen Wessels perturbation vocabulary changed")
    try:
        pert_data = native.PertData(
            str(PRESCRIBE / "data") + os.sep,
            gene_set_path=str(gene_set_path),
            default_pert_graph=False,
        )
        for gene in expected_genes:
            pert_data.gene2go.setdefault(gene, set())
        pert_data.load(data_path=str(DATA_ROOT))
        pert_data.prepare_split(
            split="custom",
            seed=MAIN_SEED,
            split_dict_path=str(
                require_data_relative_path(
                    interface["paths"]["split_pickle"], "E161 custom split pickle"
                )
            ),
        )
        pert_data.dataset_name = "wessels_e160"
        edge_index, edge_weight, pert_reindex = native.Get_Graph(
            pert_data, MAIN_SEED, overlap=True
        )
        if edge_index is not None or edge_weight is not None or pert_reindex is None:
            raise RuntimeError("Unexpected direct-adapter graph tensors/reindex")
        if int(getattr(pert_data, "nodes_num", -1)) != EXPECTED_COUNTS["selected_genes"]:
            raise RuntimeError("PRESCRIBE nodes_num differs from the frozen selected-gene axis")
        if int(getattr(pert_data, "num_pert", -1)) != len(expected_genes):
            raise RuntimeError("PRESCRIBE num_pert differs from the 27-gene frozen vocabulary")
        if sorted(map(str, np.asarray(pert_data.pert_names).tolist())) != expected_genes:
            raise RuntimeError("PRESCRIBE perturbation names differ from the frozen vocabulary")
        expected_reindex_keys = set(range(len(expected_genes))) | {-1}
        if set(pert_reindex) != expected_reindex_keys:
            raise RuntimeError("PRESCRIBE perturbation reindex keys changed")
        if any(value is None for value in pert_reindex.values()):
            raise RuntimeError("A frozen perturbation gene is absent from the selected-gene axis")
        if int(pert_reindex[-1]) != EXPECTED_COUNTS["selected_genes"]:
            raise RuntimeError("PRESCRIBE control reindex sentinel changed")
        return native, pert_data, pert_reindex
    except Exception:
        restore_import_context(native)
        raise


def flatten_graphs(pert_data: Any, conditions: list[str]) -> list[Any]:
    graphs: list[Any] = []
    for condition in conditions:
        items = pert_data.dataset_processed.get(condition)
        if not items:
            raise RuntimeError(f"No development graph for {condition}")
        graphs.extend(items)
    return graphs


def load_control_and_pca(interface: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    control_path = require_data_relative_path(
        interface["paths"]["control_prior"], "E161 train-only control prior"
    )
    pca_path = require_data_relative_path(
        interface["paths"]["pca_model"], "E161 train-only PCA model"
    )
    with np.load(control_path, allow_pickle=False) as prior:
        if set(prior.files) != {
            "control_gene_mean", "control_pca_mean", "control_pca_cov", "n_train_controls"
        }:
            raise RuntimeError("E161 control prior schema changed")
        control = np.asarray(prior["control_gene_mean"], dtype=np.float32)
        control_pca = np.asarray(prior["control_pca_mean"], dtype=np.float32)
        control_cov = np.asarray(prior["control_pca_cov"], dtype=np.float32)
        n_controls = np.asarray(prior["n_train_controls"]).reshape(-1)
    with np.load(pca_path, allow_pickle=False) as pca:
        required = {
            "model_genes", "raw_gene_indices", "mean", "components",
            "explained_variance", "explained_variance_ratio",
        }
        if set(pca.files) != required:
            raise RuntimeError("E161 PCA model schema changed")
        genes = np.asarray(pca["model_genes"]).astype(str).tolist()
        pca_mean = np.asarray(pca["mean"], dtype=np.float32)
        components = np.asarray(pca["components"], dtype=np.float32)
    if control.shape != (EXPECTED_COUNTS["selected_genes"],):
        raise RuntimeError("Control-gene prior shape changed")
    if control_pca.shape != (N_PCA,) or control_cov.shape != (N_PCA, N_PCA):
        raise RuntimeError("Control PCA prior shape changed")
    if pca_mean.shape != control.shape or components.shape != (N_PCA, len(control)):
        raise RuntimeError("PCA model dimensions changed")
    if n_controls.tolist() != [424]:
        raise RuntimeError("Train-control count changed")
    if not all(np.isfinite(value).all() for value in (control, control_pca, control_cov, pca_mean, components)):
        raise RuntimeError("Non-finite E161 control/PCA asset")
    selected_axis = require_data_relative_path(
        interface["paths"]["selected_gene_axis"], "E161 selected-gene axis"
    ).read_text(encoding="utf-8").splitlines()
    if genes != selected_axis or len(genes) != EXPECTED_COUNTS["selected_genes"]:
        raise RuntimeError("PCA model and selected-gene axis differ")
    if sha256_text("\n".join(genes) + "\n") != interface["selected_gene_order_sha256"]:
        raise RuntimeError("Selected-gene order hash changed")
    return control, pca_mean, components, genes


def exhaustive_graph_audit(
    pert_data: Any,
    split: dict[str, list[str]],
    control: np.ndarray,
) -> pd.DataFrame:
    import torch

    expected_keys = set(split["train"]) | set(split["val"])
    graph_keys = set(pert_data.dataset_processed)
    if graph_keys != expected_keys:
        raise RuntimeError(
            f"Development graph keys changed: missing={sorted(expected_keys-graph_keys)}, "
            f"extra={sorted(graph_keys-expected_keys)}"
        )
    if graph_keys & set(split["test"]):
        raise RuntimeError("Test graph exists in the E161 development cache")
    if {role: list(map(str, pert_data.set2conditions[role])) for role in ("train", "val", "test")} != split:
        raise RuntimeError("PertData custom split differs from E160/E161")
    control_tensor = torch.from_numpy(control)
    rows: list[dict[str, Any]] = []
    totals = {"train": 0, "val": 0}
    for role in ("train", "val"):
        for condition in sorted(split[role]):
            graphs = pert_data.dataset_processed[condition]
            bad = 0
            max_x_delta = 0.0
            for graph in graphs:
                keys = set(graph.keys())
                if not {"x", "y", "y_pca", "y_n", "y_d", "y_s", "pert", "pert_idx", "de_idx"}.issubset(keys):
                    bad += 1
                    continue
                x = graph.x.detach().cpu().reshape(-1)
                y = graph.y.detach().cpu().reshape(-1)
                y_pca = graph.y_pca.detach().cpu().reshape(-1)
                labels = torch.cat([
                    graph.y_n.detach().cpu().reshape(-1),
                    graph.y_d.detach().cpu().reshape(-1),
                    graph.y_s.detach().cpu().reshape(-1),
                ])
                if (
                    x.shape != control_tensor.shape
                    or y.shape != control_tensor.shape
                    or y_pca.shape != (N_PCA,)
                    or not bool(torch.isfinite(x).all())
                    or not bool(torch.isfinite(y).all())
                    or not bool(torch.isfinite(y_pca).all())
                    or str(graph.pert) != condition
                ):
                    bad += 1
                    continue
                label_ok = bool(torch.isfinite(labels).all()) if role == "train" else bool(torch.isnan(labels).all())
                if not label_ok:
                    bad += 1
                expected_perts = [gene for gene in condition.split("+") if gene != "ctrl"]
                observed_indices = np.asarray(graph.pert_idx, dtype=int).reshape(-1)
                if condition == "ctrl":
                    pert_ok = observed_indices.tolist() == [-1]
                else:
                    pert_ok = (
                        len(observed_indices) == len(expected_perts)
                        and np.all(observed_indices >= 0)
                        and np.asarray(pert_data.pert_names)[observed_indices].astype(str).tolist() == expected_perts
                    )
                if not pert_ok:
                    bad += 1
                de_idx = np.asarray(graph.de_idx, dtype=int).reshape(-1)
                de_ok = (
                    len(de_idx) == 20
                    and (
                        np.all(de_idx == -1)
                        if condition == "ctrl"
                        else (np.all(de_idx >= 0) and np.all(de_idx < len(control)))
                    )
                )
                if not de_ok:
                    bad += 1
                max_x_delta = max(
                    max_x_delta, float(torch.max(torch.abs(x - control_tensor)).item())
                )
            if bad or max_x_delta > GRAPH_X_ATOL:
                raise RuntimeError(f"Malformed development graphs for {condition}")
            rows.append({
                "condition": condition,
                "split": role,
                "n_graphs": len(graphs),
                "n_genes": len(control),
                "max_abs_x_minus_train_control": max_x_delta,
                "truth_rule": "finite_train" if role == "train" else "finite_y_y_pca_nan_rank_sentinels",
                "bad_graphs": bad,
            })
            totals[role] += len(graphs)
    if totals != {"train": EXPECTED_COUNTS["train_graphs"], "val": EXPECTED_COUNTS["val_graphs"]}:
        raise RuntimeError(f"Development graph totals changed: {totals}")
    return pd.DataFrame(rows)


def make_dev_datamodule(pert_data: Any, split: dict[str, list[str]], seed: int):
    import lightning.pytorch as L
    import torch
    from torch_geometric.loader import DataLoader

    train_graphs = flatten_graphs(pert_data, split["train"])
    val_graphs = flatten_graphs(pert_data, split["val"])
    generator = torch.Generator().manual_seed(seed)

    class DevelopmentOnlyDataModule(L.LightningDataModule):
        def train_dataloader(self):
            return DataLoader(
                train_graphs,
                batch_size=BATCH_SIZE,
                shuffle=True,
                drop_last=True,
                num_workers=0,
                generator=generator,
            )

        def val_dataloader(self):
            return DataLoader(
                val_graphs,
                batch_size=BATCH_SIZE,
                shuffle=False,
                drop_last=False,
                num_workers=0,
            )

        def test_dataloader(self):
            raise RuntimeError("E162 has no test graph, test DataLoader, or test truth")

    datamodule = DevelopmentOnlyDataModule()
    if len(datamodule.train_dataloader()) != 23 or len(datamodule.val_dataloader()) != 10:
        raise RuntimeError("Development DataLoader batch counts changed")
    try:
        datamodule.test_dataloader()
    except RuntimeError as exc:
        if "no test graph" not in str(exc):
            raise
    else:
        raise RuntimeError("E162 test_dataloader did not reject access")
    return datamodule


def native_args(seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        seed=seed,
        data_name="wessels_e160",
        backbone=None,
        batch_size=BATCH_SIZE,
        latent_dim=64,
        output_dim=N_PCA,
        flow_layers=10,
        flow_size=0.774,
        flow_n_hidden=2,
        maf_layers=10,
        budget="exp",
        bound=30,
        warmup_epochs=N_WARMUP_EPOCHS,
        warmup_lr=1e-3,
        log_prob_positive=False,
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
        load_from="",
        lr=1e-4,
        lam1=1e-7,
        scheduler="plateau",
        interval="epoch",
        change_step=2,
        reduce_rate=0.99,
        warmup_steps=0,
        warmup_max_steps=int(875 * 1.25),
        lam2=0.1,
        lam3=1e-5,
    )


def clean_hparams(module: Any) -> None:
    # Lightning keeps two stores: ``hparams`` (used in checkpoints) and the
    # private initial snapshot used by loggers.  Removing the model only from
    # ``hparams`` still makes TensorBoard serialize the full 32.8M-parameter
    # object into hparams.yaml.  Purge both stores before a Trainer sees the
    # module, then enforce the same boundary on each mapping.
    stores: list[tuple[str, Any]] = [("hparams", module.hparams)]
    initial = getattr(module, "_hparams_initial", None)
    if initial is not None:
        stores.append(("_hparams_initial", initial))
    for store_name, store in stores:
        if not hasattr(store, "items"):
            raise RuntimeError(f"Lightning {store_name} is not a mapping")
        for key in ("adata", "model"):
            store.pop(key, None)
        for key, value in store.items():
            if value.__class__.__module__.startswith("anndata"):
                raise RuntimeError(
                    f"AnnData remains in Lightning {store_name}: {key}"
                )
        if {"adata", "model"} & set(map(str, store)):
            raise RuntimeError(f"Forbidden object key remains in Lightning {store_name}")


def forbidden_checkpoint_objects(value: Any, path: str = "root") -> list[str]:
    import torch

    module = value.__class__.__module__
    if module.startswith("anndata"):
        return [f"{path}:AnnData"]
    if isinstance(value, (str, Path)):
        lowered = str(value).lower()
        forbidden = ("test_truth", "test_expression", "official_scperturb", "wesselssatija2023.h5ad")
        return [f"{path}:forbidden-path-or-truth"] if any(token in lowered for token in forbidden) else []
    issues: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            issues.extend(forbidden_checkpoint_objects(key, f"{path}.<key>"))
            issues.extend(forbidden_checkpoint_objects(item, f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            issues.extend(forbidden_checkpoint_objects(item, f"{path}[{index}]"))
    elif isinstance(value, torch.Tensor):
        return issues
    return issues


def audit_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    issues = forbidden_checkpoint_objects(payload)
    if issues:
        raise RuntimeError(f"Checkpoint contains forbidden objects: {issues[:10]}")
    hparams = payload.get("hyper_parameters", {}) if isinstance(payload, dict) else {}
    if not isinstance(hparams, dict):
        raise RuntimeError(f"Checkpoint hyper_parameters is not a mapping: {path}")
    forbidden_hparams = {"adata", "model"} & set(map(str, hparams))
    if forbidden_hparams:
        raise RuntimeError(
            f"Checkpoint retains forbidden hyperparameters {sorted(forbidden_hparams)}: {path}"
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "contains_anndata_or_test_truth_path": False,
        "hyperparameter_keys": sorted(map(str, hparams)),
    }


def atomic_torch_save(path: Path, payload: Any) -> str:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable checkpoint: {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(payload, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)
    fsync_directory(path.parent)
    return sha256_file(path)


def save_slim_checkpoint(module: Any, path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    state = {key: tensor.detach().cpu() for key, tensor in module.state_dict().items()}
    payload = {
        "schema": "safeconf_e162_locked_native_state_v1",
        "state_dict": state,
        "metadata": metadata,
    }
    atomic_torch_save(path, payload)
    return audit_checkpoint(path)


def callback_audit(trainer: Any, *, expect_early_stopping: int) -> dict[str, Any]:
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    early = [callback for callback in trainer.callbacks if isinstance(callback, EarlyStopping)]
    checkpoints = [callback for callback in trainer.callbacks if isinstance(callback, ModelCheckpoint)]
    if len(early) != expect_early_stopping or len(checkpoints) != 1:
        raise RuntimeError(
            f"Unexpected Lightning callback multiplicity: early={len(early)}, "
            f"checkpoint={len(checkpoints)}"
        )
    payload: dict[str, Any] = {
        "n_early_stopping": len(early),
        "n_model_checkpoint": len(checkpoints),
        "callback_classes": [
            f"{callback.__class__.__module__}.{callback.__class__.__name__}"
            for callback in trainer.callbacks
        ],
    }
    if early:
        callback = early[0]
        if (
            callback.monitor != "val/loss"
            or callback.mode != "min"
            or float(callback.min_delta) != -1e-3
            or int(callback.patience) != 3
        ):
            # Lightning stores min_delta with a sign determined by mode.
            raise RuntimeError("Native main EarlyStopping settings differ from the frozen contract")
        payload["early_stopping"] = {
            "monitor": callback.monitor,
            "mode": callback.mode,
            "min_delta_user_value": 1e-3,
            "min_delta_internal": float(callback.min_delta),
            "patience": int(callback.patience),
        }
    return payload


def seed_status_path(attempt: Path, seed: int) -> Path:
    return attempt / f"seed_{seed}/RUN_STATUS.json"


def build_main_module(native: SimpleNamespace, pert_data: Any, seed: int):
    args = native_args(seed)
    model = native.build_model(args, pert_data)
    module = native.MainModule(
        model=model,
        learning_rate_decay=True,
        learning_rate=args.lr,
        lam1=args.lam1,
        scheduler=args.scheduler,
        interval=args.interval,
        change_step=args.change_step,
        reduce_rate=args.reduce_rate,
        patience=3,
        warmup_steps=args.warmup_steps,
        warmup_max_steps=args.warmup_max_steps,
        lam3=args.lam3,
        lam2=args.lam2,
        save_value_only=False,
        data_name=args.data_name,
        adata=None,
    )
    clean_hparams(module)
    configured = module.configure_callbacks()
    if len(configured) != 1:
        raise RuntimeError("Native main module did not configure exactly one EarlyStopping callback")
    callback = configured[0]
    if (
        callback.monitor != "val/loss"
        or callback.mode != "min"
        or int(callback.patience) != 3
        or not math.isclose(abs(float(callback.min_delta)), 1e-3, rel_tol=0.0, abs_tol=1e-15)
    ):
        raise RuntimeError("Native configured EarlyStopping differs from contract")
    return module


def immutable_seed_checkpoint_path(attempt: Path, seed: int) -> Path:
    return LOCKED_MODEL_ROOT / attempt.name / f"seed_{seed}/E162_LOCKED_NATIVE_STATE.pt"


def seed_checkpoint_ready_path(attempt: Path, seed: int) -> Path:
    return attempt / f"seed_{seed}/CHECKPOINT_READY.json"


def recover_seed_checkpoint_publication(
    native: SimpleNamespace,
    pert_data: Any,
    attempt: Path,
    seed: int,
    gate_fingerprint: str,
) -> tuple[Any, dict[str, Any]]:
    """Roll forward an atomically prepared checkpoint across the status-write window."""
    import torch

    ready_path = seed_checkpoint_ready_path(attempt, seed)
    if not ready_path.is_file() or ready_path.is_symlink():
        raise RuntimeError(f"Seed {seed} checkpoint-ready record is missing or unsafe")
    ready = load_json(ready_path)
    required_identity = {
        "schema": "safeconf_e162_checkpoint_ready_v1",
        "experiment": "E162_wessels_prescribe_native",
        "attempt": attempt.name,
        "seed": seed,
        "gate_fingerprint_sha256": gate_fingerprint,
    }
    if any(ready.get(key) != value for key, value in required_identity.items()):
        raise RuntimeError(f"Seed {seed} checkpoint-ready identity differs")
    best_audit = ready.get("best_lightning_checkpoint")
    if not isinstance(best_audit, dict):
        raise RuntimeError(f"Seed {seed} checkpoint-ready best audit is malformed")
    best_path = Path(str(best_audit.get("path")))
    observed_best = audit_checkpoint(best_path)
    if observed_best["sha256"] != best_audit.get("sha256"):
        raise RuntimeError(f"Seed {seed} best Lightning checkpoint changed during recovery")
    best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
    if not isinstance(best_payload, dict) or "state_dict" not in best_payload:
        raise RuntimeError(f"Seed {seed} best Lightning checkpoint lacks state_dict")
    module = build_main_module(native, pert_data, seed)
    incompatible = module.load_state_dict(best_payload["state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Seed {seed} recovered best state was not exact")

    slim_path = immutable_seed_checkpoint_path(attempt, seed)
    slim_metadata = {
        "experiment": "E162_wessels_prescribe_native",
        "attempt": attempt.name,
        "seed": seed,
        "role": SEED_ROLE[seed],
        "best_val_loss": float(ready["best_val_loss"]),
        "gate_fingerprint_sha256": gate_fingerprint,
        "checkpoint_ready_sha256": sha256_file(ready_path),
        "batch_size": BATCH_SIZE,
        "accumulate_grad_batches": ACCUMULATE_GRAD_BATCHES,
        "warmup_epochs": N_WARMUP_EPOCHS,
        "main_max_epochs": N_EPOCHS,
    }
    if slim_path.exists():
        slim_audit = audit_checkpoint(slim_path)
        slim_payload = torch.load(slim_path, map_location="cpu", weights_only=False)
        if (
            not isinstance(slim_payload, dict)
            or slim_payload.get("schema") != "safeconf_e162_locked_native_state_v1"
            or slim_payload.get("metadata") != slim_metadata
        ):
            raise RuntimeError(f"Seed {seed} immutable slim checkpoint metadata differs")
        for key, tensor in best_payload["state_dict"].items():
            if key not in slim_payload["state_dict"] or not torch.equal(
                tensor.detach().cpu(), slim_payload["state_dict"][key].detach().cpu()
            ):
                raise RuntimeError(f"Seed {seed} slim checkpoint differs from best state at {key}")
        if set(slim_payload["state_dict"]) != set(best_payload["state_dict"]):
            raise RuntimeError(f"Seed {seed} slim checkpoint state keys differ")
    else:
        slim_audit = save_slim_checkpoint(module, slim_path, slim_metadata)

    checkpoint_audit = {
        "best_lightning_checkpoint": observed_best,
        "locked_slim_checkpoint": slim_audit,
        "best_val_loss": float(ready["best_val_loss"]),
        "callbacks": ready["callbacks"],
        "strict_state_reload": True,
        "best_reload_forward_max_abs_delta": ready["best_reload_forward_max_abs_delta"],
        "load_from_checkpoint_used": False,
        "checkpoint_publication_rolled_forward": True,
        "checkpoint_ready_sha256": sha256_file(ready_path),
    }
    update_status(
        seed_status_path(attempt, seed),
        phase="checkpoint_locked_before_any_label_only_forward",
        completed_at=now(),
        checkpoint_audit=checkpoint_audit,
        locked_slim_checkpoint=slim_audit,
        best_val_loss=float(ready["best_val_loss"]),
        checkpoint_publication_rolled_forward=True,
        test_label_queried=False,
        test_X_accessed=False,
        test_truth_accessed=False,
    )
    return module.cpu().eval(), checkpoint_audit


def load_locked_seed_module(
    native: SimpleNamespace,
    pert_data: Any,
    attempt: Path,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    import torch

    status = load_json(seed_status_path(attempt, seed))
    if status.get("phase") != "checkpoint_locked_before_any_label_only_forward":
        raise RuntimeError(f"Seed {seed} is not checkpoint-locked")
    path = immutable_seed_checkpoint_path(attempt, seed)
    audit = audit_checkpoint(path)
    if audit["sha256"] != status["locked_slim_checkpoint"]["sha256"]:
        raise RuntimeError(f"Seed {seed} locked checkpoint hash changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    module = build_main_module(native, pert_data, seed)
    incompatible = module.load_state_dict(payload["state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Seed {seed} strict checkpoint load was not exact")
    return module, status["checkpoint_audit"]


def train_seed(
    native: SimpleNamespace,
    pert_data: Any,
    split: dict[str, list[str]],
    control: np.ndarray,
    attempt: Path,
    seed: int,
    gate_fingerprint: str,
) -> tuple[Any, dict[str, Any]]:
    import lightning.pytorch as L
    import torch
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger

    seed_dir = attempt / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    status_path = seed_status_path(attempt, seed)
    if status_path.exists():
        prior = load_json(status_path)
        if prior.get("gate_fingerprint_sha256") != gate_fingerprint:
            raise RuntimeError(f"Seed {seed} cannot resume under a different input/source gate")
        if prior.get("phase") == "checkpoint_locked_before_any_label_only_forward":
            return load_locked_seed_module(native, pert_data, attempt, seed)
        if str(prior.get("phase", "")).startswith(TERMINAL_PREFIXES):
            raise RuntimeError(f"Seed {seed} has terminal status and cannot be overwritten")
    else:
        atomic_json(status_path, {
            "experiment": "E162_wessels_prescribe_native",
            "attempt": attempt.name,
            "seed": seed,
            "role": SEED_ROLE[seed],
            "phase": "seed_claimed_no_training",
            "started_at": now(),
            "gate_fingerprint_sha256": gate_fingerprint,
            "batch_size": BATCH_SIZE,
            "accumulate_grad_batches": ACCUMULATE_GRAD_BATCHES,
            "warmup_epochs": N_WARMUP_EPOCHS,
            "main_max_epochs": N_EPOCHS,
            "test_label_queried": False,
            "test_X_accessed": False,
            "test_truth_accessed": False,
            "interrupted_resume_bitwise_identity_claimed": False,
        }, replace=False)

    ready_path = seed_checkpoint_ready_path(attempt, seed)
    slim_path = immutable_seed_checkpoint_path(attempt, seed)
    if ready_path.exists():
        return recover_seed_checkpoint_publication(
            native, pert_data, attempt, seed, gate_fingerprint
        )
    if slim_path.exists():
        raise RuntimeError(
            f"Seed {seed} has an immutable slim checkpoint without CHECKPOINT_READY provenance"
        )

    seed_everything(seed, workers=True)
    datamodule = make_dev_datamodule(pert_data, split, seed)
    args = native_args(seed)
    model = native.build_model(args, pert_data)
    warmup_state = seed_dir / "checkpoints/warmup_complete_model_state.pt"
    warmup_last = seed_dir / "checkpoints/warmup/last.ckpt"
    warmup_state.parent.mkdir(parents=True, exist_ok=True)
    if warmup_state.exists():
        status = load_json(status_path)
        observed_warmup_hash = sha256_file(warmup_state)
        if status.get("warmup_state_sha256") is None:
            if status.get("phase") != "warmup_started":
                raise RuntimeError(f"Seed {seed} orphan warmup state cannot be rolled forward")
            update_status(
                status_path,
                phase="warmup_complete",
                warmup_state_sha256=observed_warmup_hash,
                warmup_publication_rolled_forward=True,
                interrupted_resume_bitwise_identity_claimed=False,
            )
        elif status.get("warmup_state_sha256") != observed_warmup_hash:
            raise RuntimeError(f"Seed {seed} warmup state is not registered for recovery")
        model.load_state_dict(torch.load(warmup_state, map_location="cpu", weights_only=True), strict=True)
    else:
        warmup = native.WarmupModule(
            model,
            learning_rate=args.warmup_lr,
            early_stopping=False,
            stage="warmup",
            log_prob_positive=False,
        )
        clean_hparams(warmup)
        checkpoint = ModelCheckpoint(
            dirpath=str(warmup_last.parent),
            save_last=True,
            save_top_k=0,
            every_n_epochs=1,
        )
        trainer = Trainer(
            deterministic=True,
            callbacks=[checkpoint],
            max_epochs=N_WARMUP_EPOCHS,
            devices=1,
            accelerator="gpu",
            logger=TensorBoardLogger(str(seed_dir / "logs"), name="warmup"),
            log_every_n_steps=1,
            check_val_every_n_epoch=1,
            accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
        )
        pre_warmup_phase = load_json(status_path).get("phase")
        resume = str(warmup_last) if warmup_last.exists() else None
        if resume is not None and pre_warmup_phase != "warmup_started":
            raise RuntimeError(f"Seed {seed} warmup last.ckpt is not eligible for recovery")
        update_status(status_path, phase="warmup_started", model_training_started=True)
        if resume is not None:
            update_status(status_path, resumed_from_warmup_last_sha256=sha256_file(warmup_last))
        trainer.fit(model=warmup, datamodule=datamodule, ckpt_path=resume)
        callback_audit(trainer, expect_early_stopping=0)
        state = {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()}
        atomic_torch_save(warmup_state, state)
        update_status(
            status_path,
            phase="warmup_complete",
            warmup_state_sha256=sha256_file(warmup_state),
            interrupted_resume_bitwise_identity_claimed=False,
        )
        del warmup, trainer
        gc.collect()
        torch.cuda.empty_cache()

    module = native.MainModule(
        model=model,
        learning_rate_decay=True,
        learning_rate=args.lr,
        lam1=args.lam1,
        scheduler=args.scheduler,
        interval=args.interval,
        change_step=args.change_step,
        reduce_rate=args.reduce_rate,
        patience=3,
        warmup_steps=args.warmup_steps,
        warmup_max_steps=args.warmup_max_steps,
        lam3=args.lam3,
        lam2=args.lam2,
        save_value_only=False,
        data_name=args.data_name,
        adata=None,
    )
    clean_hparams(module)
    configured = module.configure_callbacks()
    if len(configured) != 1 or configured[0].monitor != "val/loss":
        raise RuntimeError("Native main EarlyStopping configuration changed")
    main_dir = seed_dir / "checkpoints/main"
    checkpoint = ModelCheckpoint(
        dirpath=str(main_dir),
        filename="best-{epoch:02d}",
        auto_insert_metric_name=False,
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        every_n_epochs=1,
    )
    trainer = Trainer(
        deterministic=True,
        callbacks=[checkpoint],
        max_epochs=N_EPOCHS,
        devices=1,
        accelerator="gpu",
        logger=TensorBoardLogger(str(seed_dir / "logs"), name="main"),
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
    )
    last_path = main_dir / "last.ckpt"
    prior_phase = load_json(status_path).get("phase")
    resume = str(last_path) if last_path.exists() else None
    if resume is not None and prior_phase != "native_training_started":
        raise RuntimeError(f"Seed {seed} orphan main last.ckpt cannot be resumed")
    update_status(
        status_path,
        phase="native_training_started",
        last_invocation_prior_phase=prior_phase,
        resumed_from_main_last_sha256=sha256_file(last_path) if resume else None,
    )
    trainer.fit(model=module, datamodule=datamodule, ckpt_path=resume)
    callback_settings = callback_audit(trainer, expect_early_stopping=1)
    if not checkpoint.best_model_path or checkpoint.best_model_score is None:
        raise RuntimeError(f"Seed {seed} has no best val/loss checkpoint")
    best_score_tensor = checkpoint.best_model_score.detach().cpu()
    if not bool(torch.isfinite(best_score_tensor).all()):
        raise RuntimeError(f"Seed {seed} best validation loss is non-finite")
    best_path = Path(checkpoint.best_model_path)
    best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
    if not isinstance(best_payload, dict) or "state_dict" not in best_payload:
        raise RuntimeError(f"Seed {seed} best Lightning checkpoint lacks state_dict")
    incompatible = module.load_state_dict(best_payload["state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Seed {seed} best checkpoint strict load was not exact")
    module.eval()

    # Rebuild the same architecture and strictly reload plain state; never call
    # Lightning load_from_checkpoint.  Confirm state and forward identity.
    rebuilt = build_main_module(native, pert_data, seed)
    incompatible = rebuilt.load_state_dict(best_payload["state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Seed {seed} rebuilt strict state load was not exact")
    for key, tensor in module.state_dict().items():
        if not torch.equal(tensor.detach().cpu(), rebuilt.state_dict()[key].detach().cpu()):
            raise RuntimeError(f"Seed {seed} rebuilt state differs at {key}")
    device = torch.device("cuda:0")
    module.to(device).eval()
    rebuilt.to(device).eval()
    equivalence_conditions = sorted(
        split["val"], key=lambda value: sha256_text(f"E162|best-reload|{seed}|{value}")
    )[:N_EQUIVALENCE_TASKS]
    query = label_only_batch(equivalence_conditions, control).to(device)
    fields_original = posterior_fields(module, query)
    fields_rebuilt = posterior_fields(rebuilt, query)
    reload_deltas = {
        name: float(np.max(np.abs(np.asarray(left) - np.asarray(right))))
        for name, left, right in zip(
            ("prediction", "raw_log_prob", "epistemic", "aleatoric"),
            fields_original,
            fields_rebuilt,
        )
    }
    if not np.isfinite(list(reload_deltas.values())).all() or max(reload_deltas.values()) != 0.0:
        raise RuntimeError(f"Seed {seed} best-checkpoint rebuild forward is not exact")
    del module, query, fields_original, fields_rebuilt
    gc.collect()
    torch.cuda.empty_cache()
    rebuilt.cpu().eval()

    lightning_audit = audit_checkpoint(best_path)
    ready_payload = {
        "schema": "safeconf_e162_checkpoint_ready_v1",
        "experiment": "E162_wessels_prescribe_native",
        "attempt": attempt.name,
        "seed": seed,
        "gate_fingerprint_sha256": gate_fingerprint,
        "best_val_loss": float(best_score_tensor.item()),
        "best_lightning_checkpoint": lightning_audit,
        "callbacks": callback_settings,
        "best_reload_forward_max_abs_delta": reload_deltas,
    }
    atomic_json(ready_path, ready_payload, replace=False)
    slim_metadata = {
        "experiment": "E162_wessels_prescribe_native",
        "attempt": attempt.name,
        "seed": seed,
        "role": SEED_ROLE[seed],
        "best_val_loss": float(best_score_tensor.item()),
        "gate_fingerprint_sha256": gate_fingerprint,
        "checkpoint_ready_sha256": sha256_file(ready_path),
        "batch_size": BATCH_SIZE,
        "accumulate_grad_batches": ACCUMULATE_GRAD_BATCHES,
        "warmup_epochs": N_WARMUP_EPOCHS,
        "main_max_epochs": N_EPOCHS,
    }
    slim_audit = save_slim_checkpoint(
        rebuilt,
        slim_path,
        slim_metadata,
    )
    checkpoint_audit = {
        "best_lightning_checkpoint": lightning_audit,
        "locked_slim_checkpoint": slim_audit,
        "best_val_loss": float(best_score_tensor.item()),
        "callbacks": callback_settings,
        "strict_state_reload": True,
        "best_reload_forward_max_abs_delta": reload_deltas,
        "load_from_checkpoint_used": False,
        "checkpoint_publication_rolled_forward": False,
        "checkpoint_ready_sha256": sha256_file(ready_path),
    }
    update_status(
        status_path,
        phase="checkpoint_locked_before_any_label_only_forward",
        completed_at=now(),
        checkpoint_audit=checkpoint_audit,
        locked_slim_checkpoint=slim_audit,
        best_val_loss=float(best_score_tensor.item()),
        checkpoint_publication_rolled_forward=False,
        interrupted_resume_bitwise_identity_claimed=False,
        test_label_queried=False,
        test_X_accessed=False,
        test_truth_accessed=False,
    )
    return rebuilt, checkpoint_audit


def label_only_batch(conditions: list[str], control: np.ndarray):
    import torch
    from torch_geometric.data import Batch, Data

    graphs = [
        Data(x=torch.from_numpy(control.copy()).float().unsqueeze(1), pert=str(condition))
        for condition in conditions
    ]
    for graph in graphs:
        if set(graph.keys()) != {"x", "pert"}:
            raise RuntimeError("Label-only source graph contains an unexpected field")
    batch = Batch.from_data_list(graphs)
    forbidden_fields = {
        "y", "y_pca", "y_n", "y_d", "y_s", "de_idx", "n_test_cells",
        "cell_count", "test_truth", "test_expression", "error",
    }
    truth_keys = sorted(forbidden_fields & set(batch.keys()))
    truth_attributes = sorted(
        key for key in forbidden_fields if getattr(batch, key, None) is not None
    )
    if truth_keys or truth_attributes:
        raise RuntimeError("Label-only batch unexpectedly contains truth")
    allowed_batch_fields = {"x", "pert", "batch", "ptr"}
    if not set(batch.keys()).issubset(allowed_batch_fields):
        raise RuntimeError(
            f"Label-only batch has fields outside its allowlist: "
            f"{sorted(set(batch.keys()) - allowed_batch_fields)}"
        )
    observed = [str(value) for value in batch.pert]
    if observed != list(map(str, conditions)):
        raise RuntimeError("Label-only perturbation order changed during batching")
    if int(batch.num_graphs) != len(conditions) or tuple(batch.x.shape) != (
        len(conditions) * len(control), 1
    ):
        raise RuntimeError("Label-only batch shape changed")
    return batch


def posterior_fields(module: Any, batch: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import torch

    module.eval()
    with torch.inference_mode():
        posterior, raw_log_prob = module.model(batch)
        prediction = posterior.maximum_a_posteriori().loc
        epistemic, aleatoric = module._calculate_unc(posterior, log_prob=raw_log_prob)
    return (
        prediction.detach().cpu().numpy(),
        raw_log_prob.detach().cpu().numpy(),
        epistemic.detach().cpu().numpy(),
        aleatoric.detach().cpu().numpy(),
    )


def forward_equivalence_audit(
    module: Any,
    pert_data: Any,
    validation_conditions: list[str],
    control: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    import torch
    from torch_geometric.data import Batch

    conditions = sorted(
        validation_conditions,
        key=lambda value: sha256_text(
            "E162|Wessels|forward-equivalence|20260714|v1\t" + str(value)
        ),
    )[:N_EQUIVALENCE_TASKS]
    native_batch = Batch.from_data_list(
        [pert_data.dataset_processed[condition][0] for condition in conditions]
    )
    query_batch = label_only_batch(conditions, control)
    device = torch.device("cuda:0")
    module.to(device).eval()
    native_fields = posterior_fields(module, native_batch.to(device))
    query_fields = posterior_fields(module, query_batch.to(device))
    rows: list[dict[str, Any]] = []
    names = ("prediction", "raw_log_prob", "epistemic", "aleatoric")
    for index, condition in enumerate(conditions):
        row: dict[str, Any] = {"seed": seed, "condition": condition}
        for name, native_values, query_values in zip(names, native_fields, query_fields):
            left = np.asarray(native_values[index])
            right = np.asarray(query_values[index])
            if not np.isfinite(left).all() or not np.isfinite(right).all():
                raise RuntimeError(f"Seed {seed} non-finite {name} in forward-equivalence audit")
            row[f"max_abs_delta_{name}"] = float(np.max(np.abs(left - right)))
        rows.append(row)
    frame = pd.DataFrame(rows)
    delta_columns = [column for column in frame if column.startswith("max_abs_delta_")]
    if len(frame) != N_EQUIVALENCE_TASKS or frame[delta_columns].to_numpy(float).max() > FORWARD_EQUIVALENCE_ATOL:
        raise RuntimeError(f"Seed {seed} label-only/native forward equivalence failed")
    module.cpu()
    del native_batch, query_batch, native_fields, query_fields
    gc.collect()
    torch.cuda.empty_cache()
    return frame


def exact_n_unique_vectors(values: np.ndarray) -> int:
    contiguous = np.ascontiguousarray(values)
    return int(np.unique(contiguous, axis=0).shape[0])


def nondegeneracy_gate(table: pd.DataFrame, *, split: str) -> dict[str, Any]:
    expected_rows = 24 if split == "validation" else 48
    required_unique = 12 if split == "validation" else 24
    raw = table["raw_log_prob"].to_numpy(float)
    prediction = table[[f"predicted_pca_{index}" for index in range(N_PCA)]].to_numpy(float)
    raw_finite = bool(np.isfinite(raw).all())
    prediction_finite = bool(np.isfinite(prediction).all())
    raw_unique = int(np.unique(raw).size) if raw_finite else 0
    prediction_unique = exact_n_unique_vectors(prediction) if prediction_finite else 0
    raw_std = float(np.std(raw, ddof=1)) if raw_finite and len(raw) > 1 else float("nan")
    coordinate_std = (
        np.std(prediction, axis=0, ddof=1)
        if prediction_finite and len(prediction) > 1
        else np.full(N_PCA, np.nan)
    )
    passed = bool(
        len(table) == expected_rows
        and table["condition"].nunique() == expected_rows
        and raw_finite
        and raw_unique >= required_unique
        and raw_std > RAW_LOG_STD_THRESHOLD
        and prediction.shape == (expected_rows, N_PCA)
        and prediction_finite
        and prediction_unique >= required_unique
        and np.any(coordinate_std > RAW_LOG_STD_THRESHOLD)
    )
    return {
        "split": split,
        "passed": passed,
        "n_rows": len(table),
        "required_rows": expected_rows,
        "raw_log_prob_all_finite": raw_finite,
        "raw_log_prob_exact_unique": raw_unique,
        "required_minimum_exact_unique": required_unique,
        "raw_log_prob_sample_std_ddof1": raw_std,
        "prediction_shape": list(prediction.shape),
        "prediction_all_finite": prediction_finite,
        "prediction_exact_unique_vectors": prediction_unique,
        "prediction_coordinate_sample_std_ddof1": coordinate_std.tolist(),
        "prediction_any_coordinate_std_gt_1e_minus_6": bool(
            np.any(coordinate_std > RAW_LOG_STD_THRESHOLD)
        ),
    }


def estimability_record(
    values: np.ndarray,
    *,
    seed: int,
    split: str,
    score: str,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=float).reshape(-1)
    finite = bool(np.isfinite(values).all())
    unique = int(np.unique(values).size) if finite else 0
    std = float(np.std(values, ddof=1)) if finite and len(values) > 1 else float("nan")
    estimable = bool(finite and unique >= 2 and std > BASELINE_STD_THRESHOLD)
    return {
        "seed": seed,
        "split": split,
        "score": score,
        "n": len(values),
        "all_finite": finite,
        "exact_unique": unique,
        "sample_std_ddof1": std,
        "estimable": estimable,
        "failure_code": "" if estimable else "constant_or_nonfinite_baseline",
        "downstream_statistic": "eligible" if estimable else "NA",
    }


def query_label_only_scores(
    module: Any,
    conditions: list[str],
    control: np.ndarray,
    pca_mean: np.ndarray,
    components: np.ndarray,
    genes: list[str],
    *,
    seed: int,
    split: str,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    import torch

    device = torch.device("cuda:0")
    module.to(device).eval()
    batch = label_only_batch(conditions, control).to(device)
    prediction, raw_log_prob, epistemic, aleatoric = posterior_fields(module, batch)
    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.shape != (len(conditions), N_PCA):
        raise RuntimeError(f"Seed {seed} {split} prediction shape changed: {prediction.shape}")

    def scalar_output(name: str, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape == (len(conditions),):
            return array
        if array.shape == (len(conditions), 1):
            return array[:, 0]
        raise RuntimeError(f"Seed {seed} {split} {name} shape changed: {array.shape}")

    raw_log_prob = scalar_output("raw_log_prob", raw_log_prob)
    epistemic = scalar_output("epistemic", epistemic)
    aleatoric = scalar_output("aleatoric", aleatoric)
    official = 2.0 * epistemic + aleatoric
    reconstructed = prediction @ components.astype(np.float64) + pca_mean.astype(np.float64)
    magnitude = np.sqrt(np.mean((reconstructed - control.astype(np.float64)[None, :]) ** 2, axis=1))
    rows: list[dict[str, Any]] = []
    gene_hash = sha256_text("\n".join(genes) + "\n")
    for index, condition in enumerate(conditions):
        row: dict[str, Any] = {
            "seed": seed,
            "seed_role": SEED_ROLE[seed],
            "split": split,
            "condition": condition,
            "query_has_test_expression": False,
            "query_has_y": False,
            "query_has_y_pca": False,
            "raw_log_prob": float(raw_log_prob[index]),
            "epistemic_confidence": float(epistemic[index]),
            "aleatoric_confidence": float(aleatoric[index]),
            "official_combined_confidence": float(official[index]),
            "predicted_magnitude_rms": float(magnitude[index]),
            "selected_gene_order_sha256": gene_hash,
        }
        row.update({
            f"predicted_pca_{dimension}": float(prediction[index, dimension])
            for dimension in range(N_PCA)
        })
        rows.append(row)
    table = pd.DataFrame(rows)
    if table["condition"].tolist() != list(conditions):
        raise RuntimeError(f"Seed {seed} {split} label-only row order changed")
    gate = nondegeneracy_gate(table, split=split)
    estimability = pd.DataFrame([
        estimability_record(
            table[column].to_numpy(float), seed=seed, split=split, score=score
        )
        for column, score in (
            ("official_combined_confidence", "official_combined_confidence"),
            ("predicted_magnitude_rms", "predicted_magnitude_rms"),
        )
    ])
    module.cpu()
    del batch
    gc.collect()
    torch.cuda.empty_cache()
    return table, gate, estimability


class FrozenGateFailure(RuntimeError):
    def __init__(self, phase: str, message: str):
        super().__init__(message)
        self.phase = phase


def attempt_input_tables(preflight: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_rows = pd.DataFrame(preflight["prescribe_sources"])
    if source_rows["actual_sha256"].isna().any():
        raise RuntimeError("Formal source gate did not hash every source/weight payload")
    input_rows: list[dict[str, Any]] = []
    for row in preflight["git"]["files"]:
        input_rows.append({
            "role": f"git_{row['role']}",
            "path": row["path"],
            "sha256": row["working_sha256"],
            "bytes": (ROOT / row["path"]).stat().st_size,
        })
    input_rows.extend([
        {
            "role": "E160_split_json",
            "path": str(E160_SPLIT),
            "sha256": preflight["e160_split_sha256"],
            "bytes": E160_SPLIT.stat().st_size,
        },
        {
            "role": "E161_interface_v2",
            "path": str(INTERFACE),
            "sha256": preflight["e161_interface_sha256"],
            "bytes": INTERFACE.stat().st_size,
        },
    ])
    for row in preflight["e161_assets"]:
        if not row["payload_hashed"] or row["actual_sha256"] != row["expected_sha256"]:
            raise RuntimeError(f"Formal E161 asset was not fully hash-gated: {row['relative_path']}")
        input_rows.append({
            "role": "E161_development_asset",
            "path": str(DATA_ROOT / row["relative_path"]),
            "sha256": row["actual_sha256"],
            "bytes": row["bytes"],
        })
    return (
        source_rows.sort_values("relative_path").reset_index(drop=True),
        pd.DataFrame(input_rows).sort_values(["role", "path"]).reset_index(drop=True),
    )


def claim_or_verify_attempt(attempt: Path, fresh: bool, preflight: dict[str, Any], gpu_index: int) -> Path:
    status_path = attempt / "RUN_STATUS.json"
    if fresh:
        atomic_json(status_path, {
            "experiment": "E162_wessels_prescribe_native",
            "attempt": attempt.name,
            "phase": "attempt_claimed_inputs_not_loaded",
            "started_at": now(),
            "git_head": preflight["git"]["git_head"],
            "gpu_physical_index": gpu_index,
            "gpu_internal_index": 0,
            "gate_fingerprint_sha256": preflight["gate_fingerprint_sha256"],
            "seeds": list(SEEDS),
            "main_seed": MAIN_SEED,
            "batch_size": BATCH_SIZE,
            "accumulate_grad_batches": ACCUMULATE_GRAD_BATCHES,
            "warmup_epochs": N_WARMUP_EPOCHS,
            "main_max_epochs": N_EPOCHS,
            "listmle_limitation": LISTMLE_LIMITATION,
            "model_training_started": False,
            "validation_label_queries_started": False,
            "test_label_queries_started": False,
            "test_X_accessed": False,
            "test_truth_accessed": False,
            "test_endpoint_computed": False,
            "interrupted_resume_bitwise_identity_claimed": False,
        }, replace=False)
    else:
        status = load_json(status_path)
        if status.get("gate_fingerprint_sha256") != preflight["gate_fingerprint_sha256"]:
            raise RuntimeError("Interrupted E162 attempt cannot resume under changed inputs/sources")
        if status.get("git_head") != preflight["git"]["git_head"]:
            raise RuntimeError("Interrupted E162 attempt cannot resume under a different Git HEAD")
        if int(status.get("gpu_physical_index")) != gpu_index:
            raise RuntimeError("Interrupted E162 attempt cannot move to a different physical GPU")
    return status_path


def write_test_query_event(
    attempt: Path,
    preflight: dict[str, Any],
    checkpoint_audits: dict[int, dict[str, Any]],
    main_validation_gate_path: Path,
    main_validation_gate: dict[str, Any],
    source_manifest_path: Path,
    input_manifest_path: Path,
) -> dict[str, Any]:
    path = attempt / "TEST_LABEL_QUERY_EVENT.json"
    if main_validation_gate.get("passed") is not True:
        raise RuntimeError("Cannot authorize test-label forward before the main validation gate passes")
    if load_json(main_validation_gate_path) != main_validation_gate:
        raise RuntimeError("Main validation gate file differs from the in-memory gate")
    core = {
        "experiment": "E162_wessels_prescribe_native",
        "attempt": attempt.name,
        "event": "irreversible_test_label_only_query_authorized",
        "main_seed": MAIN_SEED,
        "main_validation_gate_passed": True,
        "test_conditions_sha256": sha256_text("\n".join(preflight["split"]["test"]) + "\n"),
        "n_test_conditions": len(preflight["split"]["test"]),
        "main_locked_checkpoint_sha256": checkpoint_audits[MAIN_SEED]["locked_slim_checkpoint"]["sha256"],
        "main_validation_gate_sha256": sha256_file(main_validation_gate_path),
        "main_validation_gate": main_validation_gate,
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "gate_fingerprint_sha256": preflight["gate_fingerprint_sha256"],
        "query_contains_test_expression": False,
        "query_contains_test_truth": False,
        "test_X_accessed": False,
        "test_truth_accessed": False,
    }
    if path.exists():
        existing = load_json(path)
        for key, value in core.items():
            if existing.get(key) != value:
                raise RuntimeError("Existing test-label query event differs from current frozen gate")
        return existing
    payload = dict(core)
    payload["written_before_first_test_label_forward_at"] = now()
    atomic_json(path, payload, replace=False)
    return payload


def artifact_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    return {
        "path": path.relative_to(relative_to).as_posix() if relative_to else str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def fsync_tree(root: Path) -> None:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"Release staging is not a regular directory: {root}")
    paths = list(root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise RuntimeError("Symlink rejected inside E162 release staging")
    for path in sorted(value for value in paths if value.is_file()):
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    directories = sorted(
        [root, *(value for value in paths if value.is_dir())],
        key=lambda value: len(value.parts),
        reverse=True,
    )
    for directory in directories:
        fsync_directory(directory)


def update_release_transaction(path: Path, transaction: dict[str, Any], phase: str) -> dict[str, Any]:
    updated = dict(transaction)
    updated["phase"] = phase
    updated["updated_at"] = now()
    history = list(updated.get("phase_history", []))
    history.append({"phase": phase, "at": updated["updated_at"]})
    updated["phase_history"] = history
    atomic_json(path, updated)
    return updated


def validate_release_tree(
    root: Path, allowed: set[str], transaction: dict[str, Any]
) -> pd.DataFrame:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"E162 release tree is not a regular directory: {root}")
    paths = list(root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise RuntimeError("Symlink rejected inside E162 release tree")
    observed = {
        path.relative_to(root).as_posix() for path in paths if path.is_file()
    }
    if observed != allowed:
        raise RuntimeError(f"E162 release allowlist mismatch: {observed ^ allowed}")
    sentinel = load_json(root / "RELEASE_TRANSACTION_SENTINEL.json")
    if sentinel != {
        "schema": "safeconf_e162_release_sentinel_v1",
        "attempt": transaction["attempt"],
        "transaction_id": transaction["transaction_id"],
        "gate_fingerprint_sha256": transaction["gate_fingerprint_sha256"],
    }:
        raise RuntimeError("E162 release sentinel differs from its transaction")
    manifest_path = root / "OUTPUT_MANIFEST.csv"
    manifest = pd.read_csv(manifest_path)
    if list(manifest.columns) != ["path", "bytes", "sha256"]:
        raise RuntimeError("E162 OUTPUT_MANIFEST schema changed")
    if manifest["path"].astype(str).duplicated().any():
        raise RuntimeError("E162 OUTPUT_MANIFEST contains duplicate paths")
    expected_manifest_paths = allowed - {"OUTPUT_MANIFEST.csv"}
    if set(manifest["path"].astype(str)) != expected_manifest_paths:
        raise RuntimeError("E162 OUTPUT_MANIFEST path set changed")
    for row in manifest.itertuples(index=False):
        path = root / str(row.path)
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != int(row.bytes)
            or sha256_file(path) != str(row.sha256)
        ):
            raise RuntimeError(f"E162 release artifact changed: {row.path}")
    return pd.DataFrame(
        [
            artifact_record(path, relative_to=root)
            for path in sorted(root.rglob("*"))
            if path.is_file()
        ]
    )


def publish_release(
    attempt: Path,
    preflight: dict[str, Any],
    checkpoint_audits: dict[int, dict[str, Any]],
    validation_gates: dict[int, dict[str, Any]],
    test_gates: dict[int, dict[str, Any] | None],
    sensitivity_validation_failures: list[int],
    sensitivity_test_gate_failures: list[int],
    source_manifest_path: Path,
    input_manifest_path: Path,
    graph_audit_path: Path,
    table_paths: list[Path],
) -> dict[str, Any]:
    staging = attempt / ".release.staging"
    release = attempt / "release"
    transaction_path = attempt / "RELEASE_TRANSACTION.json"
    sources = [source_manifest_path, input_manifest_path, graph_audit_path, *table_paths]
    source_names = [path.name for path in sources]
    if len(source_names) != len(set(source_names)):
        raise RuntimeError("Duplicate E162 release table basename")
    allowed = {
        "E162_E163_INTERFACE.json",
        "RUN_STATUS.json",
        "TEST_LABEL_QUERY_EVENT.json",
        "RELEASE_TRANSACTION_SENTINEL.json",
        "OUTPUT_MANIFEST.csv",
        *{f"tables/{name}" for name in source_names},
    }
    transaction_core = {
        "schema": "safeconf_e162_release_transaction_v1",
        "experiment": "E162_wessels_prescribe_native",
        "attempt": attempt.name,
        "gate_fingerprint_sha256": preflight["gate_fingerprint_sha256"],
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "test_label_query_event_sha256": sha256_file(attempt / "TEST_LABEL_QUERY_EVENT.json"),
        "main_validation_gate_sha256": sha256_bytes(
            canonical_json_bytes(validation_gates[MAIN_SEED])
        ),
        "main_test_gate_sha256": sha256_bytes(canonical_json_bytes(test_gates[MAIN_SEED])),
        "source_artifacts": [artifact_record(path) for path in sources],
        "allowed_files": sorted(allowed),
    }
    if transaction_path.exists():
        transaction = load_json(transaction_path)
        if any(transaction.get(key) != value for key, value in transaction_core.items()):
            raise RuntimeError("Existing E162 release transaction differs from current artifacts")
    else:
        transaction = {
            **transaction_core,
            "transaction_id": uuid.uuid4().hex,
            "phase": "building",
            "created_at": now(),
            "phase_history": [{"phase": "building", "at": now()}],
        }
        atomic_json(transaction_path, transaction, replace=False)

    if release.exists():
        release_manifest = validate_release_tree(release, allowed, transaction)
        if transaction.get("phase") != "complete":
            transaction = update_release_transaction(transaction_path, transaction, "complete")
        release_status = load_json(release / "RUN_STATUS.json")
        return {
            "release": str(release),
            "release_files": release_manifest.to_dict("records"),
            "release_status": release_status,
            "output_manifest_sha256": sha256_file(release / "OUTPUT_MANIFEST.csv"),
            "e162_e163_interface_sha256": sha256_file(release / "E162_E163_INTERFACE.json"),
            "transaction_id": transaction["transaction_id"],
            "publication_rolled_forward": True,
        }

    if staging.exists():
        if transaction.get("phase") == "ready_to_publish":
            validate_release_tree(staging, allowed, transaction)
            fsync_tree(staging)
            staging.replace(release)
            fsync_directory(attempt)
            release_manifest = validate_release_tree(release, allowed, transaction)
            transaction = update_release_transaction(transaction_path, transaction, "complete")
            return {
                "release": str(release),
                "release_files": release_manifest.to_dict("records"),
                "release_status": load_json(release / "RUN_STATUS.json"),
                "output_manifest_sha256": sha256_file(release / "OUTPUT_MANIFEST.csv"),
                "e162_e163_interface_sha256": sha256_file(release / "E162_E163_INTERFACE.json"),
                "transaction_id": transaction["transaction_id"],
                "publication_rolled_forward": True,
            }
        if transaction.get("phase") != "building":
            raise RuntimeError("E162 release staging exists in an invalid transaction phase")
        staging_paths = list(staging.rglob("*"))
        if any(path.is_symlink() for path in staging_paths):
            raise RuntimeError("Unsafe symlink in incomplete E162 release staging")
        observed = {
            path.relative_to(staging).as_posix()
            for path in staging_paths
            if path.is_file()
        }
        if not observed.issubset(allowed):
            raise RuntimeError("Unexpected file in incomplete E162 release staging")
        sentinel_path = staging / "RELEASE_TRANSACTION_SENTINEL.json"
        if observed and not sentinel_path.is_file():
            raise RuntimeError("Incomplete E162 staging has files but no transaction sentinel")
        if sentinel_path.is_file():
            sentinel = load_json(sentinel_path)
            if sentinel.get("transaction_id") != transaction["transaction_id"]:
                raise RuntimeError("Incomplete E162 staging belongs to another transaction")
        shutil.rmtree(staging)
        fsync_directory(attempt)

    (staging / "tables").mkdir(parents=True)
    sentinel_payload = {
        "schema": "safeconf_e162_release_sentinel_v1",
        "attempt": attempt.name,
        "transaction_id": transaction["transaction_id"],
        "gate_fingerprint_sha256": preflight["gate_fingerprint_sha256"],
    }
    atomic_json(staging / "RELEASE_TRANSACTION_SENTINEL.json", sentinel_payload, replace=False)
    for source in sources:
        shutil.copyfile(source, staging / "tables" / source.name)
    shutil.copyfile(attempt / "TEST_LABEL_QUERY_EVENT.json", staging / "TEST_LABEL_QUERY_EVENT.json")

    checkpoint_interface: dict[str, Any] = {}
    score_interface: dict[str, Any] = {}
    for seed in SEEDS:
        checkpoint_interface[str(seed)] = {
            "role": SEED_ROLE[seed],
            "best_val_loss": checkpoint_audits[seed]["best_val_loss"],
            "best_lightning_checkpoint": checkpoint_audits[seed]["best_lightning_checkpoint"],
            "locked_slim_checkpoint": checkpoint_audits[seed]["locked_slim_checkpoint"],
        }
        validation_path = staging / "tables" / f"E162_VALIDATION_LABEL_ONLY_SCORES_SEED{seed}.csv"
        test_path = staging / "tables" / f"E162_TEST_LABEL_ONLY_SCORES_SEED{seed}.csv"
        score_interface[str(seed)] = {
            "validation": artifact_record(validation_path, relative_to=staging),
            "test": artifact_record(test_path, relative_to=staging) if test_path.exists() else None,
        }
    interface = {
        "schema": "safeconf_e162_to_e163_v1",
        "experiment": "E162_wessels_prescribe_native",
        "attempt": attempt.name,
        "release_transaction_id": transaction["transaction_id"],
        "main_seed": MAIN_SEED,
        "main_seed_role": "unique_primary_model",
        "sensitivity_seeds": [3408, 3409],
        "sensitivity_role": "training_randomness_only_no_main_model_rescue",
        "e161_interface_schema": preflight["interface"]["schema"],
        "gate_fingerprint_sha256": preflight["gate_fingerprint_sha256"],
        "selected_gene_order_sha256": preflight["interface"]["selected_gene_order_sha256"],
        "pca_model_sha256": preflight["interface"]["asset_sha256"]["TRAIN_ONLY_PCA_MODEL.npz"],
        "control_prior_sha256": preflight["interface"]["asset_sha256"]["TRAIN_ONLY_CONTROL_PRIOR.npz"],
        "graph_cache_sha256": preflight["interface"]["asset_sha256"]["data_pyg/cell_graphs.pkl"],
        "split_pickle_sha256": preflight["interface"]["asset_sha256"][f"set2conditions_{MAIN_SEED}.pkl"],
        "checkpoints": checkpoint_interface,
        "label_only_score_tables": score_interface,
        "validation_nondegeneracy_gates": {str(k): v for k, v in validation_gates.items()},
        "test_nondegeneracy_gates": {str(k): v for k, v in test_gates.items()},
        "main_validation_gate_passed": validation_gates[MAIN_SEED]["passed"],
        "main_test_gate_passed": bool(test_gates[MAIN_SEED] and test_gates[MAIN_SEED]["passed"]),
        "sensitivity_test_not_queried_due_to_validation_gate": sensitivity_validation_failures,
        "sensitivity_test_gate_failed_after_query": sensitivity_test_gate_failures,
        "test_label_query_event_sha256": sha256_file(attempt / "TEST_LABEL_QUERY_EVENT.json"),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "output_manifest_path": "OUTPUT_MANIFEST.csv",
        "raw_h5ad_opened": False,
        "test_X_accessed": False,
        "test_truth_accessed": False,
        "test_endpoint_computed": False,
        "test_graphs": 0,
        "eligible_for_E163": True,
        "interrupted_resume_bitwise_identity_claimed": False,
        "listmle_limitation": LISTMLE_LIMITATION,
    }
    atomic_json(staging / "E162_E163_INTERFACE.json", interface, replace=False)
    release_status = {
        "experiment": "E162_wessels_prescribe_native",
        "attempt": attempt.name,
        "release_transaction_id": transaction["transaction_id"],
        "phase": (
            "complete_main_label_only_gates_passed_with_sensitivity_gate_failures"
            if sensitivity_validation_failures or sensitivity_test_gate_failures
            else "complete_all_checkpoint_and_label_only_gates_passed"
        ),
        "completed_at": now(),
        "main_seed": MAIN_SEED,
        "sensitivity_validation_gate_failures": sensitivity_validation_failures,
        "sensitivity_test_gate_failures_after_query": sensitivity_test_gate_failures,
        "main_test_gate_passed": True,
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "output_manifest_path": "OUTPUT_MANIFEST.csv",
        "raw_h5ad_opened": False,
        "test_X_accessed": False,
        "test_truth_accessed": False,
        "test_endpoint_computed": False,
        "test_graphs": 0,
        "eligible_for_E163": True,
        "interrupted_resume_bitwise_identity_claimed": False,
        "listmle_limitation": LISTMLE_LIMITATION,
    }
    atomic_json(staging / "RUN_STATUS.json", release_status, replace=False)
    manifest = pd.DataFrame(
        [
            artifact_record(path, relative_to=staging)
            for path in sorted(staging.rglob("*"))
            if path.is_file() and path.name != "OUTPUT_MANIFEST.csv"
        ]
    )
    atomic_csv(staging / "OUTPUT_MANIFEST.csv", manifest, replace=False)
    validate_release_tree(staging, allowed, transaction)
    fsync_tree(staging)
    transaction = update_release_transaction(transaction_path, transaction, "ready_to_publish")
    staging.replace(release)
    fsync_directory(attempt)
    release_manifest = validate_release_tree(release, allowed, transaction)
    transaction = update_release_transaction(transaction_path, transaction, "complete")
    return {
        "release": str(release),
        "release_files": release_manifest.to_dict("records"),
        "release_status": release_status,
        "output_manifest_sha256": sha256_file(release / "OUTPUT_MANIFEST.csv"),
        "e162_e163_interface_sha256": sha256_file(release / "E162_E163_INTERFACE.json"),
        "transaction_id": transaction["transaction_id"],
        "publication_rolled_forward": False,
    }


def run_formal(preflight: dict[str, Any], gpu_index: int) -> Path:
    import torch

    attempt, fresh = select_append_only_attempt(preflight["gate_fingerprint_sha256"])
    status_path = claim_or_verify_attempt(attempt, fresh, preflight, gpu_index)
    source_frame, input_frame = attempt_input_tables(preflight)
    source_manifest_path = attempt / "E162_SOURCE_MANIFEST.csv"
    input_manifest_path = attempt / "E162_INPUT_MANIFEST.csv"
    source_manifest_hash = atomic_csv(source_manifest_path, source_frame, replace=False)
    input_manifest_hash = atomic_csv(input_manifest_path, input_frame, replace=False)
    update_status(
        status_path,
        phase="all_input_source_and_asset_hashes_verified",
        source_manifest_sha256=source_manifest_hash,
        input_manifest_sha256=input_manifest_hash,
        development_h5ad_opened=False,
        development_graph_cache_opened=False,
        test_X_accessed=False,
        test_truth_accessed=False,
    )

    native: SimpleNamespace | None = None
    modules: dict[int, Any] = {}
    try:
        native, pert_data, _pert_reindex = load_development_pertdata(preflight)
        control, pca_mean, components, genes = load_control_and_pca(preflight["interface"])
        graph_audit = exhaustive_graph_audit(pert_data, preflight["split"], control)
        graph_audit_path = attempt / "E162_DEVELOPMENT_GRAPH_AUDIT.csv"
        atomic_csv(graph_audit_path, graph_audit, replace=False)
        update_status(
            status_path,
            phase="development_adapter_and_exhaustive_graph_audit_passed",
            development_h5ad_opened=True,
            development_graph_cache_opened=True,
            n_train_graphs=EXPECTED_COUNTS["train_graphs"],
            n_validation_graphs=EXPECTED_COUNTS["val_graphs"],
            n_test_graphs=0,
            train_batches_per_epoch=23,
            validation_batches_per_epoch=10,
            test_dataloader_rejected=True,
            test_X_accessed=False,
            test_truth_accessed=False,
        )

        checkpoint_audits: dict[int, dict[str, Any]] = {}
        for seed in SEEDS:
            update_status(status_path, phase=f"training_seed_{seed}", model_training_started=True)
            module, audit = train_seed(
                native,
                pert_data,
                preflight["split"],
                control,
                attempt,
                seed,
                preflight["gate_fingerprint_sha256"],
            )
            modules[seed] = module
            checkpoint_audits[seed] = audit
        update_status(
            status_path,
            phase="all_three_seed_checkpoints_locked_before_label_only_forward",
            checkpoint_audits={str(key): value for key, value in checkpoint_audits.items()},
            test_label_queries_started=False,
            test_X_accessed=False,
            test_truth_accessed=False,
        )

        locked_dir = attempt / "locked"
        locked_dir.mkdir(exist_ok=True)
        validation_gates: dict[int, dict[str, Any]] = {}
        validation_gate_paths: dict[int, Path] = {}
        validation_estimability: list[pd.DataFrame] = []
        table_paths: list[Path] = []
        update_status(status_path, phase="validation_label_only_gates_started", validation_label_queries_started=True)
        for seed in SEEDS:
            equivalence = forward_equivalence_audit(
                modules[seed], pert_data, preflight["split"]["val"], control, seed
            )
            equivalence_path = locked_dir / f"E162_FORWARD_EQUIVALENCE_SEED{seed}.csv"
            atomic_csv(equivalence_path, equivalence, replace=False)
            table_paths.append(equivalence_path)
            validation_table, gate, estimability = query_label_only_scores(
                modules[seed],
                preflight["split"]["val"],
                control,
                pca_mean,
                components,
                genes,
                seed=seed,
                split="validation",
            )
            validation_path = locked_dir / f"E162_VALIDATION_LABEL_ONLY_SCORES_SEED{seed}.csv"
            gate_path = locked_dir / f"E162_VALIDATION_NONDEGENERACY_GATE_SEED{seed}.json"
            atomic_csv(validation_path, validation_table, replace=False)
            atomic_json(gate_path, gate, replace=False)
            table_paths.extend([validation_path, gate_path])
            validation_gates[seed] = gate
            validation_gate_paths[seed] = gate_path
            validation_estimability.append(estimability)
            update_status(
                seed_status_path(attempt, seed),
                phase="checkpoint_locked_before_any_label_only_forward",
                validation_forward_equivalence_sha256=sha256_file(equivalence_path),
                validation_label_only_scores_sha256=sha256_file(validation_path),
                validation_nondegeneracy_gate=gate,
                test_label_queried=False,
                test_X_accessed=False,
                test_truth_accessed=False,
            )
        update_status(
            status_path,
            phase="validation_label_only_gates_locked",
            validation_nondegeneracy_gates={str(key): value for key, value in validation_gates.items()},
            test_label_queries_started=False,
            test_X_accessed=False,
            test_truth_accessed=False,
        )
        if not validation_gates[MAIN_SEED]["passed"]:
            raise FrozenGateFailure(
                "failed_main_validation_nondegeneracy_gate_no_test_label_query",
                "Main seed 3407 failed the frozen validation non-degeneracy gate; test labels remain unqueried",
            )

        event = write_test_query_event(
            attempt,
            preflight,
            checkpoint_audits,
            validation_gate_paths[MAIN_SEED],
            validation_gates[MAIN_SEED],
            source_manifest_path,
            input_manifest_path,
        )
        update_status(
            status_path,
            phase="main_test_label_only_query_authorized",
            test_label_queries_started=True,
            test_label_query_event_sha256=sha256_file(attempt / "TEST_LABEL_QUERY_EVENT.json"),
            test_X_accessed=False,
            test_truth_accessed=False,
        )
        test_gates: dict[int, dict[str, Any] | None] = {seed: None for seed in SEEDS}
        test_estimability: list[pd.DataFrame] = []
        main_table, main_gate, main_estimability = query_label_only_scores(
            modules[MAIN_SEED],
            preflight["split"]["test"],
            control,
            pca_mean,
            components,
            genes,
            seed=MAIN_SEED,
            split="test",
        )
        main_path = locked_dir / f"E162_TEST_LABEL_ONLY_SCORES_SEED{MAIN_SEED}.csv"
        main_gate_path = locked_dir / f"E162_TEST_NONDEGENERACY_GATE_SEED{MAIN_SEED}.json"
        atomic_csv(main_path, main_table, replace=False)
        atomic_json(main_gate_path, main_gate, replace=False)
        table_paths.extend([main_path, main_gate_path])
        test_gates[MAIN_SEED] = main_gate
        test_estimability.append(main_estimability)
        update_status(
            seed_status_path(attempt, MAIN_SEED),
            phase="checkpoint_locked_before_any_label_only_forward",
            test_label_queried=True,
            test_label_only_scores_sha256=sha256_file(main_path),
            test_nondegeneracy_gate=main_gate,
            test_X_accessed=False,
            test_truth_accessed=False,
        )
        if not main_gate["passed"]:
            raise FrozenGateFailure(
                "failed_main_test_nondegeneracy_gate_no_E163_unseal",
                "Main seed 3407 failed the frozen 48-label non-degeneracy gate; E163 remains sealed",
            )

        sensitivity_validation_failures: list[int] = []
        sensitivity_test_gate_failures: list[int] = []
        for seed in (3408, 3409):
            if not validation_gates[seed]["passed"]:
                sensitivity_validation_failures.append(seed)
                continue
            table, gate, estimability = query_label_only_scores(
                modules[seed],
                preflight["split"]["test"],
                control,
                pca_mean,
                components,
                genes,
                seed=seed,
                split="test",
            )
            path = locked_dir / f"E162_TEST_LABEL_ONLY_SCORES_SEED{seed}.csv"
            gate_path = locked_dir / f"E162_TEST_NONDEGENERACY_GATE_SEED{seed}.json"
            atomic_csv(path, table, replace=False)
            atomic_json(gate_path, gate, replace=False)
            table_paths.extend([path, gate_path])
            test_gates[seed] = gate
            test_estimability.append(estimability)
            if not gate["passed"]:
                sensitivity_test_gate_failures.append(seed)
            update_status(
                seed_status_path(attempt, seed),
                phase="checkpoint_locked_before_any_label_only_forward",
                test_label_queried=True,
                test_label_only_scores_sha256=sha256_file(path),
                test_nondegeneracy_gate=gate,
                test_X_accessed=False,
                test_truth_accessed=False,
            )

        estimability = pd.concat(
            [*validation_estimability, *test_estimability], ignore_index=True
        ).sort_values(["split", "seed", "score"]).reset_index(drop=True)
        estimability_path = locked_dir / "E162_BASELINE_ESTIMABILITY.csv"
        atomic_csv(estimability_path, estimability, replace=False)
        table_paths.append(estimability_path)
        checkpoint_path = locked_dir / "E162_CHECKPOINT_AUDIT.json"
        atomic_json(
            checkpoint_path,
            {"seeds": {str(key): value for key, value in checkpoint_audits.items()}},
            replace=False,
        )
        table_paths.append(checkpoint_path)
        release_audit = publish_release(
            attempt,
            preflight,
            checkpoint_audits,
            validation_gates,
            test_gates,
            sensitivity_validation_failures,
            sensitivity_test_gate_failures,
            source_manifest_path,
            input_manifest_path,
            graph_audit_path,
            table_paths,
        )
        final_phase = release_audit["release_status"]["phase"]
        update_status(
            status_path,
            phase=final_phase,
            completed_at=now(),
            release=release_audit,
            validation_nondegeneracy_gates={str(key): value for key, value in validation_gates.items()},
            test_nondegeneracy_gates={str(key): value for key, value in test_gates.items()},
            main_test_gate_passed=True,
            sensitivity_validation_gate_failures=sensitivity_validation_failures,
            sensitivity_test_gate_failures_after_query=sensitivity_test_gate_failures,
            test_label_query_event=event,
            test_X_accessed=False,
            test_truth_accessed=False,
            test_endpoint_computed=False,
            raw_h5ad_opened=False,
            n_test_graphs=0,
            interrupted_resume_bitwise_identity_claimed=False,
        )
        return status_path
    except Exception as exc:
        published_release = attempt / "release"
        if published_release.is_dir() and not published_release.is_symlink():
            # The release tree is the durable commit point.  A crash between its
            # atomic rename and the parent-status update must be resumed and
            # validated, rather than converted into a terminal failed attempt.
            phase = "release_published_parent_status_pending"
        else:
            phase = (
                exc.phase
                if isinstance(exc, FrozenGateFailure)
                else "failed_no_test_truth_access_requires_audit"
            )
        timestamp_update = (
            {"interrupted_at": now()}
            if phase == "release_published_parent_status_pending"
            else {"failed_at": now()}
        )
        update_status(
            status_path,
            phase=phase,
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            traceback=traceback.format_exc(),
            test_label_queries_started=(attempt / "TEST_LABEL_QUERY_EVENT.json").exists(),
            test_X_accessed=False,
            test_truth_accessed=False,
            test_endpoint_computed=False,
            raw_h5ad_opened=False,
            n_test_graphs=0,
            interrupted_resume_bitwise_identity_claimed=False,
            **timestamp_update,
        )
        raise
    finally:
        for module in modules.values():
            try:
                module.cpu()
            except Exception:
                pass
        modules.clear()
        gc.collect()
        torch.cuda.empty_cache()
        if native is not None:
            restore_import_context(native)


def main() -> None:
    args = parse_args()
    if args.gpu_index < 0:
        raise RuntimeError("--gpu-index must be a non-negative physical GPU index")
    if args.mode == "formal":
        # This must precede every torch, Lightning, or Step2_train import.
        os.environ.update({
            "CUDA_VISIBLE_DEVICES": str(args.gpu_index),
            "CUDA_LAUNCH_BLOCKING": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "WANDB_MODE": "offline",
        })
    preflight = metadata_preflight(formal=args.mode == "formal")
    if args.mode == "preflight":
        print(json.dumps(preflight_output(preflight), ensure_ascii=False, indent=2))
        return

    import torch

    if torch.version.cuda != "11.8":
        raise RuntimeError(
            f"E162 requires the frozen PyTorch CUDA runtime 11.8, found {torch.version.cuda}"
        )
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"CUDA visibility gate requires exactly one internal GPU, found {torch.cuda.device_count()}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the formal E162 runtime")
    torch.cuda.set_device(0)
    if torch.cuda.current_device() != 0:
        raise RuntimeError("Formal E162 must use internal cuda:0")
    status_path = run_formal(preflight, args.gpu_index)
    print(status_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
