#!/usr/bin/env python3
"""E196: audit CPA 0.8.8 native latent support distance.

This runner is intentionally post-truth and inference-only.  It rebuilds the
exact E83/E84 CPA structures, strictly loads the frozen module state, recreates
the already-saved pseudo-test predictions, and then computes distances from
each query condition to explicitly enumerated *training* conditions.

The CPA score audited here is a latent support/OOD distance.  It is not a
decoder variance, a calibrated probability, or an error bound.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.util
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/实验结果/E196_cpa_native_support_distance_20260730"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
ANALYSIS_FREEZE = OUT / "ANALYSIS_FREEZE.md"
CODE_LOCK = OUT / "E196_CODE_LOCK.json"
RUNNER = Path(__file__).resolve()

E81 = ROOT / "docs/实验结果/E81_sciplex_cartesian_contract_20260712"
E84 = ROOT / "docs/实验结果/E84_cpa_rdkit_cartesian_formal_20260712"
E94 = ROOT / "docs/实验结果/E94_cpa_pinned_reproduction_20260712"
E83_SCRIPT = ROOT / "tools/scripts/run_e83_cpa_rdkit_pilot.py"
E84_SCRIPT = ROOT / "tools/scripts/run_e84_cpa_rdkit_cartesian_formal.py"
CPA_SOURCE = Path("/home/yyf/archive/external/cpa")
CPA_RUNTIME_PYTHON = Path("/home/yyf/.conda/envs/cpa_runtime_env/bin/python")
SOURCE_H5AD = Path(
    "/home/yyf/data/singlecell_perturbation_atlas/extra_official/"
    "cellular_context_generalization/sciplex3.h5ad"
)
SMILES = Path("/home/yyf/archive/external/chemCPA/embeddings/trapnell_drugs_smiles.csv")

EVIDENCE_LABEL = "POSTTRUTH_DIRECT_COMPETITOR_AUDIT"
CPA_SOURCE_COMMIT = "fbd7c0250edc23eff003a10c99655579c53afd63"
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20260730
COVERAGES = np.round(np.arange(0.50, 1.001, 0.05), 2)
BUDGETS = (0.10, 0.20, 0.30)
PREDICTION_TOLERANCE = 1e-5
UNIQUE_ROUND_DECIMALS = 12

LOCKED_COMMON = {
    E81 / "tables/E81_SPLIT_MANIFEST.csv":
        "00aaed01fece99b55595c982faa75f6b134b4de277a2415a982036eb7ce427e2",
    E81 / "tables/E81_GENE_PANEL.csv":
        "71cb9dd8d16897f2fa8ebbcdf6cab0981a080847602ba7c3f006be2c3308280e",
    E83_SCRIPT:
        "b748f8a455698873c89af9635d6ff5ed0824747f0eb21770869667233853ad72",
    E94 / "tables/E83_TASK_SCORES.csv":
        "3a71241a9c0effe9eba4776e660054d84c87fe2ea6ff8b0f7b29e68f31f1b1a7",
}

FORMAL_STATE_HASHES = {
    "E81_r1_p25": "cf8c449a08ba6bcdc9aecdf89d0429410aacdea078013e614196ea7059329540",
    "E81_r1_p50": "1c79f7976d9a3230a1c88e5192167855b861129c06bf9493f7a2fec99fc727de",
    "E81_r2_p25": "830a317f6cff5fc7496d894f2c6dbfc37c2041f6d3eea1e3cb015ef974b5a2bb",
    "E81_r2_p50": "5b1c5e33ae174e234696069c7e1bd50e93d78361b52bf30dde42a54d6040ce62",
    "E81_r2_p75": "d33e6b3c8c9258c8c355cce6f8a9ef979b3317db86f5ade23d5673052aaafaea",
    "E81_r3_p25": "fd6aa4b87fc5fc84535dacc70ad7e362f598947691cc76e759ec094931be0f18",
    "E81_r3_p50": "0710eee8153d534f72208e83f0db23a58b7ebe12a570f7e10fc008643f98b2e0",
    "E81_r3_p75": "f45131b07e593da1dfd19fd36fc87c1e92f9c834bb265ac70525438241208f7b",
}

E94_STATE_HASH = "7e457786638cb85a183a82e7ce7b6442ed1c63dc76fccacfa5d28c3866b9d51d"
FORMAL_MANIFESTS = tuple(FORMAL_STATE_HASHES)
REFERENCE_SETS = (
    "all_explicit_train_conditions",
    "official_gt30_train_conditions",
    "perturbed_train_only",
)
DISTANCE_SCORES = (
    "native_cosine_distance",
    "native_euclidean_distance",
)
SAME_OUTCOME_SCORES = DISTANCE_SCORES + ("predicted_magnitude_cpa",)


class AuditFailure(RuntimeError):
    """Fail-closed E196 audit error."""


def utcish_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(repr(tuple(array.shape)).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def sha256_strings(values: Iterable[object]) -> str:
    normalized = "\n".join(sorted(str(value) for value in values)) + "\n"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def lock_key(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def artifact_set_digest(paths: Iterable[Path]) -> Tuple[str, int, int]:
    unique = sorted(set(Path(path).resolve() for path in paths), key=lock_key)
    rows = [f"{lock_key(path)}\t{sha256_file(path)}" for path in unique]
    payload = "\n".join(rows) + "\n"
    return (
        hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        len(unique),
        sum(path.stat().st_size for path in unique),
    )


def cpa_runtime_source_files() -> List[Path]:
    return sorted((CPA_SOURCE / "cpa").rglob("*.py"))


def state_digest(state: Mapping[str, Any]) -> str:
    """Hash tensor values without relying on nondeterministic torch serialization."""
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        array = tensor.numpy()
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(repr(tuple(array.shape)).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def stable_int(*parts: object) -> int:
    text = "\0".join(map(str, parts)).encode()
    return int(hashlib.sha256(text).hexdigest()[:16], 16)


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        rendered = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                rendered.append("NA" if not math.isfinite(float(value)) else f"{float(value):.{digits}f}")
            else:
                rendered.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def load_npz_keys(path: Path) -> List[str]:
    with np.load(path, allow_pickle=False) as archive:
        return [str(key) for key in archive.files]


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            str(key): np.asarray(archive[key], dtype=np.float32)
            for key in archive.files
        }


def git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def git_status_short(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def require_head_tracked(path: Path) -> None:
    """Require the worktree file to equal the blob recorded at repository HEAD."""
    relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
    result = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"HEAD:{relative}"],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise AuditFailure(f"formal implementation is not tracked at HEAD: {relative}")
    head_digest = hashlib.sha256(result.stdout).hexdigest()
    observed_digest = sha256_file(path)
    if head_digest != observed_digest:
        raise AuditFailure(
            f"worktree differs from HEAD for {relative}: "
            f"HEAD {head_digest}, worktree {observed_digest}"
        )


def manifest_specs() -> List[Dict[str, Any]]:
    specs = [
        {
            "manifest_id": "E81_r1_p75",
            "evidence_role": "development_adapter",
            "is_formal": False,
            "seed": 20260712,
            "epochs": 10,
            "directory": E94,
            "expected_state_sha256": E94_STATE_HASH,
            "expected_tasks": 59,
        }
    ]
    for index, manifest_id in enumerate(FORMAL_MANIFESTS):
        directory = E84 / "manifests" / manifest_id
        specs.append(
            {
                "manifest_id": manifest_id,
                "evidence_role": "formal_manifest",
                "is_formal": True,
                "seed": 20268400 + index,
                "epochs": 20,
                "directory": directory,
                "expected_state_sha256": FORMAL_STATE_HASHES[manifest_id],
                "expected_tasks": None,
            }
        )
    return specs


def required_manifest_paths(spec: Mapping[str, Any]) -> Dict[str, Path]:
    directory = Path(spec["directory"])
    return {
        "state": directory / "raw_cpa/cpa_rdkit_state.pt",
        "prediction_manifest": directory / "tables/E83_PREDICTION_MANIFEST.csv",
        "prediction_array": directory / "arrays/E83_CPA_PREDICTED_EFFECTS.npz",
        "task_scores": directory / "tables/E83_TASK_SCORES.csv",
        "predict_status": directory / "PREDICT_STATUS.json",
        "run_status": directory / "RUN_STATUS.json",
    }


def package_version(distribution: str, module_name: Optional[str] = None) -> str:
    try:
        from importlib import metadata

        return metadata.version(distribution)
    except Exception:
        if module_name is None:
            return "NOT_INSTALLED"
        try:
            module = importlib.import_module(module_name)
            return str(getattr(module, "__version__", "UNKNOWN"))
        except Exception:
            return "NOT_INSTALLED"


def runtime_environment(command: str) -> pd.DataFrame:
    rows = [
        ("generated_at", utcish_now()),
        ("command", command),
        ("safeconf_repository_head", git_head(ROOT)),
        ("python_executable", sys.executable),
        ("python_version", platform.python_version()),
        ("platform", platform.platform()),
        ("hostname", socket.gethostname()),
        ("cpu_count", str(os.cpu_count())),
        ("cpa_source_commit", git_head(CPA_SOURCE)),
        ("cpa_source_worktree_status", git_status_short(CPA_SOURCE)),
        (
            "cpa_covariate_embedding_access",
            (
                "direct module.covars_embeddings lookup using the frozen "
                "model.covars_encoder category id; CPA 0.8.8's scalar wrapper "
                "creates a 3-D AnnData input in this runtime"
            ),
        ),
    ]
    packages = [
        ("cpa-tools", "cpa"),
        ("torch", "torch"),
        ("scanpy", "scanpy"),
        ("anndata", "anndata"),
        ("scvi-tools", "scvi"),
        ("rdkit-pypi", "rdkit"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        ("scikit-learn", "sklearn"),
        ("matplotlib", "matplotlib"),
    ]
    for distribution, module_name in packages:
        rows.append((f"package::{distribution}", package_version(distribution, module_name)))
    try:
        import torch

        rows.extend(
            [
                ("torch_cuda_available", str(bool(torch.cuda.is_available()))),
                ("torch_cuda_version", str(torch.version.cuda)),
                ("gpu_count", str(torch.cuda.device_count())),
                (
                    "gpu_names",
                    "; ".join(
                        torch.cuda.get_device_name(index)
                        for index in range(torch.cuda.device_count())
                    ),
                ),
            ]
        )
    except Exception as exc:
        rows.append(("torch_runtime_error", repr(exc)))
    return pd.DataFrame(rows, columns=["key", "value"])


def add_hash_row(
    rows: List[Dict[str, Any]],
    path: Path,
    role: str,
    expected: Optional[str] = None,
) -> None:
    if not path.exists():
        raise AuditFailure(f"missing required input: {path}")
    observed = sha256_file(path)
    matched: Any = ""
    if expected is not None:
        matched = observed == expected
        if not matched:
            raise AuditFailure(
                f"SHA-256 mismatch for {path}: expected {expected}, observed {observed}"
            )
    rows.append(
        {
            "artifact_scope": "input",
            "artifact_role": role,
            "path": str(path),
            "bytes": path.stat().st_size,
            "expected_sha256": expected or "",
            "observed_sha256": observed,
            "hash_locked": expected is not None,
            "hash_match": matched,
        }
    )


TASK_METADATA_COLUMNS = (
    "manifest_id",
    "task_key",
    "context",
    "perturbation_key",
    "dose_key",
    "quadrant",
    "n_cells",
)


def normalized_task_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in TASK_METADATA_COLUMNS if column not in frame]
    if missing:
        raise AuditFailure(f"task metadata columns are missing: {missing}")
    result = frame[list(TASK_METADATA_COLUMNS)].copy()
    for column in (
        "manifest_id",
        "task_key",
        "context",
        "perturbation_key",
        "quadrant",
    ):
        result[column] = result[column].astype(str)
    result["dose_key"] = result["dose_key"].astype(float)
    result["n_cells"] = result["n_cells"].astype(int)
    return result.sort_values("task_key").reset_index(drop=True)


def preflight(command: str) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """Verify immutable assets, source version, environment, and task joins."""
    expected_python = CPA_RUNTIME_PYTHON.resolve()
    observed_python = Path(sys.executable).resolve()
    if observed_python != expected_python:
        raise AuditFailure(
            "E196 must run in cpa_runtime_env; use "
            f"{CPA_RUNTIME_PYTHON} (observed {sys.executable})"
        )

    import cpa
    import torch

    if str(getattr(cpa, "__version__", "")) != "0.8.8":
        raise AuditFailure(f"CPA version must be 0.8.8, observed {getattr(cpa, '__version__', None)}")
    observed_commit = git_head(CPA_SOURCE)
    if observed_commit != CPA_SOURCE_COMMIT:
        raise AuditFailure(
            f"CPA source commit mismatch: expected {CPA_SOURCE_COMMIT}, observed {observed_commit}"
        )

    if not CODE_LOCK.exists():
        raise AuditFailure(f"missing formal implementation lock: {CODE_LOCK}")
    code_lock = json.loads(CODE_LOCK.read_text())
    required_lock_keys = {
        "runner_sha256",
        "analysis_freeze_sha256",
        "source_h5ad_sha256",
        "smiles_sha256",
        "cpa_runtime_source_set_sha256",
        "cpa_runtime_source_count",
        "frozen_artifact_set_sha256",
        "frozen_artifact_count",
    }
    if not required_lock_keys.issubset(code_lock):
        missing = sorted(required_lock_keys - set(code_lock))
        raise AuditFailure(f"E196 code lock is incomplete: {missing}")
    for tracked in (RUNNER, ANALYSIS_FREEZE, CODE_LOCK):
        require_head_tracked(tracked)
    hashes: List[Dict[str, Any]] = []
    add_hash_row(hashes, CODE_LOCK, "formal_code_lock")
    add_hash_row(hashes, RUNNER, "formal_runner", code_lock["runner_sha256"])
    add_hash_row(
        hashes,
        ANALYSIS_FREEZE,
        "formal_analysis_freeze",
        code_lock["analysis_freeze_sha256"],
    )
    for path, expected in LOCKED_COMMON.items():
        add_hash_row(hashes, path, "frozen_common", expected)
    add_hash_row(
        hashes,
        SOURCE_H5AD,
        "source_h5ad",
        code_lock["source_h5ad_sha256"],
    )
    add_hash_row(hashes, SMILES, "smiles_mapping", code_lock["smiles_sha256"])
    cpa_sources = cpa_runtime_source_files()
    cpa_digest, cpa_count, cpa_bytes = artifact_set_digest(cpa_sources)
    if (
        cpa_digest != str(code_lock["cpa_runtime_source_set_sha256"])
        or cpa_count != int(code_lock["cpa_runtime_source_count"])
    ):
        raise AuditFailure(
            "CPA runtime Python source set changed: "
            f"count={cpa_count}, sha256={cpa_digest}"
        )
    for path in cpa_sources:
        add_hash_row(hashes, path, "cpa_runtime_source")
    hashes.append(
        {
            "artifact_scope": "input",
            "artifact_role": "cpa_runtime_source_set_aggregate",
            "path": "/home/yyf/archive/external/cpa/cpa/**/*.py (path + SHA-256)",
            "bytes": cpa_bytes,
            "expected_sha256": str(code_lock["cpa_runtime_source_set_sha256"]),
            "observed_sha256": cpa_digest,
            "hash_locked": True,
            "hash_match": True,
        }
    )
    add_hash_row(hashes, E84_SCRIPT, "frozen_e84_launcher")

    frozen_manifest = pd.read_csv(E81 / "tables/E81_SPLIT_MANIFEST.csv")
    if frozen_manifest.duplicated(["manifest_id", "task_key"]).any():
        raise AuditFailure("E81 manifest contains duplicated manifest/task keys")

    total_formal_tasks = 0
    frozen_artifact_paths = [E84_SCRIPT]
    for spec in manifest_specs():
        manifest_id = str(spec["manifest_id"])
        paths = required_manifest_paths(spec)
        frozen_artifact_paths.extend(paths.values())
        add_hash_row(
            hashes,
            paths["state"],
            f"{manifest_id}::cpa_state",
            str(spec["expected_state_sha256"]),
        )
        for role in (
            "prediction_manifest",
            "prediction_array",
            "task_scores",
            "predict_status",
            "run_status",
        ):
            add_hash_row(hashes, paths[role], f"{manifest_id}::{role}")

        prediction_manifest = pd.read_csv(paths["prediction_manifest"])
        split = frozen_manifest.loc[
            frozen_manifest["manifest_id"].eq(manifest_id)
            & frozen_manifest["role"].eq("test")
        ]
        expected_keys = set(split["task_key"].astype(str))
        predicted_keys = set(prediction_manifest["task_key"].astype(str))
        if (
            len(prediction_manifest) != len(predicted_keys)
            or expected_keys != predicted_keys
        ):
            raise AuditFailure(
                f"{manifest_id}: truth-free split/prediction join is not one-to-one"
            )
        try:
            pd.testing.assert_frame_equal(
                normalized_task_metadata(split),
                normalized_task_metadata(prediction_manifest),
                check_dtype=False,
                check_exact=False,
                rtol=0.0,
                atol=1e-12,
            )
        except AssertionError as exc:
            raise AuditFailure(
                f"{manifest_id}: split/prediction task metadata changed"
            ) from exc
        expected_count = spec["expected_tasks"]
        if expected_count is not None and len(expected_keys) != int(expected_count):
            raise AuditFailure(
                f"{manifest_id}: expected {expected_count} tasks, observed {len(expected_keys)}"
            )
        if bool(spec["is_formal"]):
            total_formal_tasks += len(expected_keys)

        array_keys = set(load_npz_keys(paths["prediction_array"]))
        expected_array_keys = set(prediction_manifest["predicted_effect_key"].astype(str))
        if array_keys != expected_array_keys:
            raise AuditFailure(f"{manifest_id}: prediction NPZ keys do not match manifest")

        predict_status = json.loads(paths["predict_status"].read_text())
        if int(predict_status["epochs_requested"]) != int(spec["epochs"]):
            raise AuditFailure(f"{manifest_id}: frozen epoch count changed")
        if int(predict_status["n_test_tasks"]) != len(expected_keys):
            raise AuditFailure(f"{manifest_id}: PREDICT_STATUS task count changed")
        if bool(predict_status.get("target_perturbed_truth_used_for_prediction", True)):
            raise AuditFailure(f"{manifest_id}: prediction status reports target truth use")

        try:
            state = torch.load(paths["state"], map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(paths["state"], map_location="cpu")
        if not isinstance(state, dict) or not state:
            raise AuditFailure(f"{manifest_id}: state file is not a nonempty state_dict")
        del state

    if total_formal_tasks != 629:
        raise AuditFailure(f"formal task total must be 629, observed {total_formal_tasks}")
    artifact_digest, artifact_count, artifact_bytes = artifact_set_digest(
        frozen_artifact_paths
    )
    if (
        artifact_digest != str(code_lock["frozen_artifact_set_sha256"])
        or artifact_count != int(code_lock["frozen_artifact_count"])
    ):
        raise AuditFailure(
            "frozen E84/E94 artifact set changed: "
            f"count={artifact_count}, sha256={artifact_digest}"
        )
    hashes.append(
        {
            "artifact_scope": "input",
            "artifact_role": "frozen_artifact_set_aggregate",
            "path": "E84/E94 required artifact set (path + SHA-256)",
            "bytes": artifact_bytes,
            "expected_sha256": str(code_lock["frozen_artifact_set_sha256"]),
            "observed_sha256": artifact_digest,
            "hash_locked": True,
            "hash_match": True,
        }
    )

    environment = runtime_environment(command)
    version = environment.set_index("key")["value"].to_dict()
    required_versions = {
        "package::cpa-tools": "0.8.8",
        "package::scanpy": "1.10.3",
        "package::scvi-tools": "0.20.3",
    }
    for key, expected in required_versions.items():
        if version.get(key) != expected:
            raise AuditFailure(f"environment mismatch: {key}={version.get(key)}, expected {expected}")
    return hashes, environment


def load_e83_core():
    spec = importlib.util.spec_from_file_location("e196_frozen_e83_core", E83_SCRIPT)
    if spec is None or spec.loader is None:
        raise AuditFailure("could not import frozen E83 script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_hyperparameters(seed: int) -> Dict[str, Any]:
    return {
        "use_rdkit_embeddings": True,
        "n_latent": 32,
        "recon_loss": "gauss",
        "doser_type": "linear",
        "n_hidden_encoder": 128,
        "n_layers_encoder": 2,
        "n_hidden_decoder": 128,
        "n_layers_decoder": 2,
        "use_batch_norm_encoder": True,
        "use_layer_norm_encoder": False,
        "use_batch_norm_decoder": True,
        "use_layer_norm_decoder": False,
        "dropout_rate_encoder": 0.1,
        "dropout_rate_decoder": 0.1,
        "variational": False,
        "seed": seed,
    }


def build_and_load_model(
    core: Any,
    spec: Mapping[str, Any],
    device: str,
) -> Tuple[Any, Any, pd.DataFrame, Any, Any, pd.DataFrame, str, List[str], List[str]]:
    """Use the frozen builder, instantiate the exact architecture, and load strictly."""
    import cpa
    import torch

    manifest_id = str(spec["manifest_id"])
    combined, manifest, panel, source, x, obs, train_tasks, val_tasks = core.build_cpa_adata(
        manifest_id,
        max_cells=32,
        control_cells=64,
        pseudo_cells=16,
        seed=int(spec["seed"]),
    )
    cpa.CPA.pert_encoder = None
    cpa.CPA.covars_encoder = None
    cpa.CPA.pert_smiles_map = None
    cpa.CPA.setup_anndata(
        combined,
        perturbation_key="drug_cpa",
        dosage_key="dose_cpa",
        control_group="control",
        smiles_key="smiles_cpa",
        is_count_data=False,
        categorical_covariate_keys=["cell_line"],
        max_comb_len=1,
    )
    model = cpa.CPA(
        combined,
        split_key="split_cpa",
        train_split="train",
        valid_split="valid",
        test_split="test",
        **model_hyperparameters(int(spec["seed"])),
    )
    paths = required_manifest_paths(spec)
    try:
        state = torch.load(paths["state"], map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(paths["state"], map_location="cpu")
    incompatible = model.module.load_state_dict(state, strict=True)
    missing = len(getattr(incompatible, "missing_keys", []))
    unexpected = len(getattr(incompatible, "unexpected_keys", []))
    if missing or unexpected:
        raise AuditFailure(
            f"{manifest_id}: strict state load has {missing} missing and {unexpected} unexpected keys"
        )
    model.module.to(torch.device(device))
    model.module.eval()
    if model.module.training:
        raise AuditFailure(f"{manifest_id}: model did not enter evaluation mode")
    loaded_hash = state_digest(model.module.state_dict())
    return (
        model,
        combined,
        manifest,
        source,
        x,
        obs,
        loaded_hash,
        sorted(str(task) for task in train_tasks),
        sorted(str(task) for task in val_tasks),
    )


def dense_mean(x: Any) -> np.ndarray:
    from scipy import sparse

    if sparse.issparse(x):
        return np.asarray(x.mean(axis=0)).ravel().astype(np.float32)
    return np.asarray(x, dtype=np.float32).mean(axis=0)


def reproduce_predictions(
    model: Any,
    combined: Any,
    manifest: pd.DataFrame,
    x: Any,
    obs: pd.DataFrame,
    spec: Mapping[str, Any],
) -> Tuple[float, int]:
    """Recreate the saved pseudo-test effects without reading target perturbation truth."""
    manifest_id = str(spec["manifest_id"])
    paths = required_manifest_paths(spec)
    frozen = load_npz(paths["prediction_array"])
    test_indices = np.flatnonzero(combined.obs["split_cpa"].eq("test").to_numpy())
    pseudo = combined[test_indices].copy()
    model.predict(pseudo, batch_size=128, n_samples=1)
    predicted_expression = np.asarray(pseudo.obsm["CPA_pred"], dtype=np.float32)

    controls: Dict[str, np.ndarray] = {}
    for context in sorted(manifest["context"].astype(str).unique()):
        mask = (
            obs["context"].astype(str).eq(context).to_numpy()
            & obs["perturbation"].astype(str).eq("control").to_numpy()
        )
        controls[context] = dense_mean(x[mask])

    maximum = 0.0
    compared = 0
    pseudo_obs = pseudo.obs.reset_index(drop=True)
    for task_key, group_indices in pseudo_obs.groupby("prediction_task_key").groups.items():
        positions = np.asarray(list(group_indices), dtype=int)
        context = str(pseudo_obs.loc[positions[0], "context"])
        effect = predicted_expression[positions].mean(axis=0) - controls[context]
        key = f"E83::{manifest_id}::{task_key}::CPA_RDKIT::pred"
        if key not in frozen:
            raise AuditFailure(f"{manifest_id}: missing frozen pseudo-test effect {key}")
        if effect.shape != frozen[key].shape:
            raise AuditFailure(f"{manifest_id}: pseudo-test effect shape changed for {task_key}")
        difference = float(np.max(np.abs(effect.astype(float) - frozen[key].astype(float))))
        maximum = max(maximum, difference)
        compared += 1
    if compared != len(frozen):
        raise AuditFailure(
            f"{manifest_id}: reproduced {compared} effects but frozen archive contains {len(frozen)}"
        )
    return maximum, compared


def condition_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["context"] = result["cell_line"].astype(str)
    result["drug"] = result["drug_cpa"].astype(str)
    result["dose_model"] = result["dose_cpa"].astype(float)
    return result


def condition_key(context: str, drug: str, dose: float) -> str:
    return f"{context}::{drug}::model_dose={float(dose):.12g}"


def condition_vector(
    model: Any,
    context: str,
    drug: str,
    dose_model: float,
    perturbation_cache: Dict[Tuple[str, float], np.ndarray],
    context_cache: Dict[str, np.ndarray],
) -> np.ndarray:
    pert_key = (str(drug), float(dose_model))
    if pert_key not in perturbation_cache:
        pert = model.get_pert_embeddings(
            dosage=float(dose_model),
            pert=str(drug),
        )
        perturbation_cache[pert_key] = np.asarray(pert.X, dtype=float).reshape(-1)
    if context not in context_cache:
        # CPA 0.8.8's public scalar wrapper constructs covar_ids with shape
        # (1, 1), so torch Embedding returns (1, 1, latent) and AnnData rejects
        # that 3-D array in the pinned runtime.  This is the exact underlying
        # lookup used by that wrapper, with a one-dimensional category-id
        # tensor; no parameter or category mapping is changed.
        import torch

        encoder = model.covars_encoder["cell_line"]
        if str(context) not in encoder:
            raise AuditFailure(f"unknown frozen cell_line category: {context}")
        covar_id = torch.tensor(
            [int(encoder[str(context)])],
            dtype=torch.long,
            device=model.device,
        )
        covar = model.module.covars_embeddings["cell_line"](covar_id)
        context_cache[context] = (
            covar.detach().cpu().numpy().astype(float, copy=False).reshape(-1)
        )
    vector = perturbation_cache[pert_key] + context_cache[context]
    if vector.shape != (32,) or not np.isfinite(vector).all():
        raise AuditFailure(
            f"invalid CPA condition vector for {condition_key(context, drug, dose_model)}"
        )
    return vector


def explicit_reference_conditions(combined: Any) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    obs = condition_columns(combined.obs.reset_index(drop=True))
    keys = ["context", "drug", "dose_model"]
    counts = (
        obs.groupby(keys + ["split_cpa"], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    wide = (
        counts.pivot_table(
            index=keys,
            columns="split_cpa",
            values="n_cells",
            fill_value=0,
            aggfunc="sum",
        )
        .reset_index()
    )
    wide.columns.name = None
    for split in ("train", "valid", "test"):
        if split not in wide:
            wide[split] = 0
        wide[split] = wide[split].astype(int)
    train = wide.loc[wide["train"].gt(0)].copy()
    train["is_control"] = train["drug"].eq("control")
    train["condition_key"] = [
        condition_key(row.context, row.drug, row.dose_model)
        for row in train.itertuples(index=False)
    ]
    if train.duplicated("condition_key").any():
        raise AuditFailure("explicit training conditions did not deduplicate one-to-one")
    sets = {
        "all_explicit_train_conditions": train.copy(),
        "official_gt30_train_conditions": train.loc[train["train"].gt(30)].copy(),
        "perturbed_train_only": train.loc[~train["is_control"]].copy(),
    }
    for name, frame in sets.items():
        if frame.empty:
            raise AuditFailure(f"reference set {name} is empty")
    return obs, sets


def extract_distances(
    model: Any,
    combined: Any,
    manifest: pd.DataFrame,
    spec: Mapping[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Compute truth-free native support distances and reference provenance."""
    manifest_id = str(spec["manifest_id"])
    prediction_manifest = pd.read_csv(required_manifest_paths(spec)["prediction_manifest"])
    observed, references = explicit_reference_conditions(combined)
    perturbation_cache: Dict[Tuple[str, float], np.ndarray] = {}
    context_cache: Dict[str, np.ndarray] = {}
    reference_rows: List[Dict[str, Any]] = []
    reference_arrays: Dict[str, np.ndarray] = {}

    for reference_name, reference in references.items():
        vectors = []
        for row in reference.sort_values("condition_key").itertuples(index=False):
            vector = condition_vector(
                model,
                str(row.context),
                str(row.drug),
                float(row.dose_model),
                perturbation_cache,
                context_cache,
            )
            vectors.append(vector)
            reference_rows.append(
                {
                    "manifest_id": manifest_id,
                    "evidence_role": spec["evidence_role"],
                    "reference_set": reference_name,
                    "reference_condition": row.condition_key,
                    "reference_context": row.context,
                    "reference_drug": row.drug,
                    "reference_dose_model_log10": row.dose_model,
                    "reference_is_control": bool(row.is_control),
                    "n_train_cells": int(row.train),
                    "n_valid_cells_same_condition": int(row.valid),
                    "n_test_cells_same_condition": int(row.test),
                    "reference_provenance": "split_cpa=train",
                    "embedding_l2_norm": float(np.linalg.norm(vector)),
                    "embedding_sha256": sha256_array(vector),
                }
            )
        reference_arrays[reference_name] = np.vstack(vectors)

    task_rows: List[Dict[str, Any]] = []
    test = manifest.loc[manifest["role"].eq("test")].set_index("task_key")
    for row in prediction_manifest.sort_values("task_key").itertuples(index=False):
        task_key = str(row.task_key)
        if task_key not in test.index:
            raise AuditFailure(f"{manifest_id}: prediction task absent from frozen test manifest")
        model_dose = float(np.log10(max(float(row.dose_key), 1.0)))
        query = condition_vector(
            model,
            str(row.context),
            str(row.perturbation_key),
            model_dose,
            perturbation_cache,
            context_cache,
        )
        for reference_name, reference in references.items():
            reference = reference.sort_values("condition_key").reset_index(drop=True)
            matrix = reference_arrays[reference_name]
            cosine = cosine_distances(query.reshape(1, -1), matrix)[0]
            euclidean = euclidean_distances(query.reshape(1, -1), matrix)[0]
            if not np.isfinite(cosine).all() or not np.isfinite(euclidean).all():
                raise AuditFailure(f"{manifest_id}: nonfinite distance for {task_key}")
            cosine_index = int(np.argmin(cosine))
            euclidean_index = int(np.argmin(euclidean))
            cosine_nearest = reference.iloc[cosine_index]
            euclidean_nearest = reference.iloc[euclidean_index]
            task_rows.append(
                {
                    "evidence_label": EVIDENCE_LABEL,
                    "manifest_id": manifest_id,
                    "evidence_role": spec["evidence_role"],
                    "is_formal": bool(spec["is_formal"]),
                    "task_key": task_key,
                    "context": row.context,
                    "perturbation_key": row.perturbation_key,
                    "dose_key_raw_nM": float(row.dose_key),
                    "dose_model_log10": model_dose,
                    "quadrant": row.quadrant,
                    "n_cells": int(row.n_cells),
                    "reference_set": reference_name,
                    "n_reference_conditions": len(reference),
                    "native_cosine_distance": float(cosine[cosine_index]),
                    "native_euclidean_distance": float(euclidean[euclidean_index]),
                    "nearest_cosine_condition": cosine_nearest.condition_key,
                    "nearest_cosine_context": cosine_nearest.context,
                    "nearest_cosine_drug": cosine_nearest.drug,
                    "nearest_cosine_dose_model_log10": cosine_nearest.dose_model,
                    "nearest_cosine_is_control": bool(cosine_nearest.is_control),
                    "nearest_euclidean_condition": euclidean_nearest.condition_key,
                    "nearest_euclidean_context": euclidean_nearest.context,
                    "nearest_euclidean_drug": euclidean_nearest.drug,
                    "nearest_euclidean_dose_model_log10": euclidean_nearest.dose_model,
                    "nearest_euclidean_is_control": bool(euclidean_nearest.is_control),
                    "predicted_magnitude_cpa": float(row.predicted_magnitude_cpa),
                    "predicted_effect_key": str(row.predicted_effect_key),
                    "query_embedding_l2_norm": float(np.linalg.norm(query)),
                    "query_embedding_sha256": sha256_array(query),
                }
            )

    all_train = references["all_explicit_train_conditions"]
    audit = {
        "manifest_id": manifest_id,
        "n_explicit_train_conditions": len(all_train),
        "n_reference_all": len(references["all_explicit_train_conditions"]),
        "n_reference_gt30": len(references["official_gt30_train_conditions"]),
        "n_reference_perturbed": len(references["perturbed_train_only"]),
        "reference_validation_overlap_conditions": int(
            (
                (all_train["valid"] > 0)
                | (all_train["test"] > 0)
            ).sum()
        ),
        "n_control_nearest_cosine_all": 0,
        "n_control_nearest_euclidean_all": 0,
    }
    tasks = pd.DataFrame(task_rows)
    all_rows = tasks.loc[tasks["reference_set"].eq("all_explicit_train_conditions")]
    audit["n_control_nearest_cosine_all"] = int(all_rows["nearest_cosine_is_control"].sum())
    audit["n_control_nearest_euclidean_all"] = int(all_rows["nearest_euclidean_is_control"].sum())
    return tasks, pd.DataFrame(reference_rows), audit


def join_outcomes(task_distances: pd.DataFrame, spec: Mapping[str, Any]) -> pd.DataFrame:
    manifest_id = str(spec["manifest_id"])
    scores = pd.read_csv(required_manifest_paths(spec)["task_scores"])
    columns = [
        "task_key",
        "predicted_magnitude_cpa",
        "cpa_ridge_disagreement_rmse",
        "error_cpa_rmse",
        "pair_mean_rmse",
    ]
    if any(column not in scores for column in columns):
        raise AuditFailure(f"{manifest_id}: existing task score schema is incomplete")
    numeric = scores[columns[1:]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise AuditFailure(f"{manifest_id}: outcome table contains nonfinite values")
    if (numeric.to_numpy(float) < 0).any():
        raise AuditFailure(f"{manifest_id}: RMSE/magnitude score is negative")
    if scores.duplicated("task_key").any():
        raise AuditFailure(f"{manifest_id}: duplicate task outcomes")
    expected_keys = set(task_distances["task_key"].astype(str))
    scored_keys = set(scores["task_key"].astype(str))
    if expected_keys != scored_keys or len(scores) != len(scored_keys):
        raise AuditFailure(f"{manifest_id}: outcome task-key set changed")
    preoutcome_metadata = (
        task_distances[
            [
                "manifest_id",
                "task_key",
                "context",
                "perturbation_key",
                "dose_key_raw_nM",
                "quadrant",
                "n_cells",
            ]
        ]
        .drop_duplicates("task_key")
        .rename(columns={"dose_key_raw_nM": "dose_key"})
    )
    try:
        pd.testing.assert_frame_equal(
            normalized_task_metadata(preoutcome_metadata),
            normalized_task_metadata(scores),
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=1e-12,
        )
    except AssertionError as exc:
        raise AuditFailure(
            f"{manifest_id}: outcome metadata differs from the pre-outcome stage"
        ) from exc
    if "predicted_effect_key" in scores:
        expected_effect_keys = (
            task_distances[["task_key", "predicted_effect_key"]]
            .drop_duplicates("task_key")
            .sort_values("task_key")
            .reset_index(drop=True)
        )
        observed_effect_keys = (
            scores[["task_key", "predicted_effect_key"]]
            .sort_values("task_key")
            .reset_index(drop=True)
        )
        if not expected_effect_keys.equals(observed_effect_keys):
            raise AuditFailure(f"{manifest_id}: predicted-effect key metadata changed")
    before = len(task_distances)
    joined = task_distances.merge(
        scores[columns].rename(columns={"predicted_magnitude_cpa": "frozen_magnitude_check"}),
        on="task_key",
        how="left",
        validate="many_to_one",
    )
    if len(joined) != before or joined[columns[2:]].isna().any().any():
        raise AuditFailure(f"{manifest_id}: outcome join failed")
    magnitude_difference = np.max(
        np.abs(
            joined["predicted_magnitude_cpa"].to_numpy(float)
            - joined["frozen_magnitude_check"].to_numpy(float)
        )
    )
    if magnitude_difference > 1e-10:
        raise AuditFailure(f"{manifest_id}: prediction magnitude changed ({magnitude_difference})")
    joined = joined.drop(columns="frozen_magnitude_check")
    joined["outcome_joined_after_distance"] = True
    return joined


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    first = np.asarray(x, dtype=float)
    second = np.asarray(y, dtype=float)
    keep = np.isfinite(first) & np.isfinite(second)
    if keep.sum() < 4:
        return float("nan")
    if np.unique(first[keep]).size < 2 or np.unique(second[keep]).size < 2:
        return float("nan")
    first_rank = stats.rankdata(first[keep], method="average")
    second_rank = stats.rankdata(second[keep], method="average")
    first_rank -= first_rank.mean()
    second_rank -= second_rank.mean()
    denominator = float(
        np.sqrt(np.dot(first_rank, first_rank) * np.dot(second_rank, second_rank))
    )
    if denominator <= 0:
        return float("nan")
    return float(np.dot(first_rank, second_rank) / denominator)


def stable_order(
    values: np.ndarray,
    task_keys: Sequence[str],
    score_name: str,
    descending: bool,
) -> np.ndarray:
    # The same task-key tie order is shared by every score.  Score-specific
    # tie breaking can create artificial paired routing differences.
    ties = np.asarray(
        [stable_int("E196", "shared_routing_tie", task_key) for task_key in task_keys],
        dtype=np.uint64,
    )
    primary = -np.asarray(values, dtype=float) if descending else np.asarray(values, dtype=float)
    return np.lexsort((ties, primary))


def routing_for_group(
    group: pd.DataFrame,
    score_column: str,
    target_column: str,
    score_name: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    work = group.loc[
        np.isfinite(group[score_column]) & np.isfinite(group[target_column])
    ].copy()
    if len(work) < 4:
        return [], []
    risk = work[score_column].to_numpy(float)
    error = work[target_column].to_numpy(float)
    if (
        np.unique(np.round(risk, UNIQUE_ROUND_DECIMALS)).size < 2
        or np.unique(np.round(error, UNIQUE_ROUND_DECIMALS)).size < 2
    ):
        return [], []
    keys = work["task_key"].astype(str).tolist()
    full_mean = float(error.mean())
    total_error = float(error.sum())
    high_order = stable_order(risk, keys, score_name, descending=True)
    low_order = stable_order(risk, keys, score_name, descending=False)
    oracle_order = stable_order(error, keys, "oracle_error", descending=True)

    metric_rows: List[Dict[str, Any]] = []
    for budget in BUDGETS:
        n_select = max(1, int(math.ceil(budget * len(work))))
        selected = high_order[:n_select]
        oracle = oracle_order[:n_select]
        remaining = high_order[n_select:]
        selected_mean = float(error[selected].mean())
        oracle_mean = float(error[oracle].mean())
        denominator = oracle_mean - full_mean
        metric_rows.append(
            {
                "budget": budget,
                "n_selected": n_select,
                "top_error_enrichment": selected_mean / full_mean,
                "top_total_error_capture": float(error[selected].sum() / total_error),
                "reject_remaining_error_reduction": (
                    float(1.0 - error[remaining].mean() / full_mean)
                    if len(remaining)
                    else float("nan")
                ),
                "selected_mean_error": selected_mean,
                "oracle_mean_error": oracle_mean,
                "oracle_normalized_utility": (
                    float((selected_mean - full_mean) / denominator)
                    if denominator > 1e-15
                    else float("nan")
                ),
            }
        )

    curve_rows: List[Dict[str, Any]] = []
    normalized_curve = []
    for coverage in COVERAGES:
        keep = max(1, int(math.ceil(coverage * len(work))))
        retained = float(error[low_order[:keep]].mean())
        normalized = retained / full_mean
        normalized_curve.append(normalized)
        curve_rows.append(
            {
                "coverage": coverage,
                "n_retained": keep,
                "retained_mean_error": retained,
                "normalized_retained_error": normalized,
            }
        )
    aurc = float(np.trapz(normalized_curve, COVERAGES) / (COVERAGES[-1] - COVERAGES[0]))
    # AURC has no review-budget dimension.  It is stored once on the 20%
    # routing row as a table anchor and is reported without an "@budget" label.
    for row in metric_rows:
        row["normalized_aurc_50_100"] = (
            aurc if abs(float(row["budget"]) - 0.20) < 1e-12 else np.nan
        )
    return metric_rows, curve_rows


def score_target_contracts() -> Tuple[Tuple[str, str, str], ...]:
    return (
        ("native_cosine_distance", "error_cpa_rmse", "same_CPA_predictor"),
        ("native_euclidean_distance", "error_cpa_rmse", "same_CPA_predictor"),
        ("predicted_magnitude_cpa", "error_cpa_rmse", "same_CPA_predictor_baseline"),
        (
            "cpa_ridge_disagreement_rmse",
            "pair_mean_rmse",
            "supplement_distinct_predictor_family",
        ),
    )


def grouping_scopes(frame: pd.DataFrame) -> Iterable[Tuple[str, pd.DataFrame]]:
    yield "overall", frame
    for quadrant, group in frame.groupby("quadrant", sort=True, observed=True):
        yield str(quadrant), group


def compute_statistics(
    task_distances: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    association_rows: List[Dict[str, Any]] = []
    routing_rows: List[Dict[str, Any]] = []
    curve_rows: List[Dict[str, Any]] = []
    dynamic_rows: List[Dict[str, Any]] = []

    for (manifest_id, reference_set), manifest in task_distances.groupby(
        ["manifest_id", "reference_set"], sort=True, observed=True
    ):
        evidence_role = str(manifest["evidence_role"].iloc[0])
        is_formal = bool(manifest["is_formal"].iloc[0])
        for score_name, target, outcome_contract in score_target_contracts():
            values = manifest[score_name].to_numpy(float)
            finite = values[np.isfinite(values)]
            unique = np.unique(np.round(finite, UNIQUE_ROUND_DECIMALS))
            estimability = (
                "ESTIMABLE"
                if len(finite) >= 4 and len(unique) >= 2
                else "NON_ESTIMABLE"
            )
            dynamic_rows.append(
                {
                    "manifest_id": manifest_id,
                    "evidence_role": evidence_role,
                    "reference_set": reference_set,
                    "score_name": score_name,
                    "n_tasks": len(values),
                    "n_finite": len(finite),
                    "n_nan": int(np.isnan(values).sum()),
                    "n_infinite": int(np.isinf(values).sum()),
                    "n_unique_rounded_12dp": len(unique),
                    "minimum": float(np.min(finite)) if len(finite) else np.nan,
                    "maximum": float(np.max(finite)) if len(finite) else np.nan,
                    "standard_deviation": float(np.std(finite)) if len(finite) else np.nan,
                    "estimability": estimability,
                }
            )
            for scope, group in grouping_scopes(manifest):
                scope_score = group[score_name].to_numpy(float)
                scope_target = group[target].to_numpy(float)
                scope_keep = np.isfinite(scope_score) & np.isfinite(scope_target)
                scope_score_unique = np.unique(
                    np.round(
                        scope_score[scope_keep],
                        UNIQUE_ROUND_DECIMALS,
                    )
                )
                scope_target_unique = np.unique(
                    np.round(
                        scope_target[scope_keep],
                        UNIQUE_ROUND_DECIMALS,
                    )
                )
                scope_estimability = (
                    "ESTIMABLE"
                    if scope_keep.sum() >= 4
                    and len(scope_score_unique) >= 2
                    and len(scope_target_unique) >= 2
                    else "NON_ESTIMABLE"
                )
                value = (
                    spearman(group[score_name], group[target])
                    if scope_estimability == "ESTIMABLE"
                    else float("nan")
                )
                association_rows.append(
                    {
                        "manifest_id": manifest_id,
                        "evidence_role": evidence_role,
                        "is_formal": is_formal,
                        "scope": scope,
                        "reference_set": reference_set,
                        "score_name": score_name,
                        "target_error": target,
                        "outcome_contract": outcome_contract,
                        "n_tasks": len(group),
                        "n_finite_pairs": int(scope_keep.sum()),
                        "n_unique_score_rounded_12dp": len(scope_score_unique),
                        "n_unique_target_rounded_12dp": len(scope_target_unique),
                        "spearman": value,
                        "estimability": scope_estimability,
                    }
                )
                metrics, curves = (
                    routing_for_group(group, score_name, target, score_name)
                    if scope_estimability == "ESTIMABLE"
                    else ([], [])
                )
                for row in metrics:
                    routing_rows.append(
                        {
                            "manifest_id": manifest_id,
                            "evidence_role": evidence_role,
                            "is_formal": is_formal,
                            "scope": scope,
                            "reference_set": reference_set,
                            "score_name": score_name,
                            "target_error": target,
                            "outcome_contract": outcome_contract,
                            "n_tasks": len(group),
                            **row,
                        }
                    )
                for row in curves:
                    curve_rows.append(
                        {
                            "manifest_id": manifest_id,
                            "evidence_role": evidence_role,
                            "is_formal": is_formal,
                            "scope": scope,
                            "reference_set": reference_set,
                            "score_name": score_name,
                            "target_error": target,
                            "outcome_contract": outcome_contract,
                            "n_tasks": len(group),
                            **row,
                        }
                    )
    return (
        pd.DataFrame(dynamic_rows),
        pd.DataFrame(association_rows),
        pd.DataFrame(routing_rows),
        pd.DataFrame(curve_rows),
    )


def bootstrap_mean(values: np.ndarray, seed: int, n_bootstrap: int) -> Tuple[float, float, int]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    draws = finite[
        rng.integers(0, len(finite), size=(n_bootstrap, len(finite)))
    ].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high), len(draws)


def macro_summary(
    association: pd.DataFrame,
    routing: pd.DataFrame,
    n_bootstrap: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    formal_assoc = association.loc[association["is_formal"]].copy()
    for keys, group in formal_assoc.groupby(
        [
            "scope",
            "reference_set",
            "score_name",
            "target_error",
            "outcome_contract",
        ],
        sort=True,
        observed=True,
    ):
        scope, reference_set, score_name, target, contract = keys
        values = group["spearman"].to_numpy(float)
        low, high, valid = bootstrap_mean(
            values,
            stable_int(BOOTSTRAP_SEED, "macro", *keys) % (2**32),
            n_bootstrap,
        )
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "scope": scope,
                "reference_set": reference_set,
                "score_name": score_name,
                "target_error": target,
                "outcome_contract": contract,
                "metric": "spearman",
                "n_manifests_total": group["manifest_id"].nunique(),
                "n_manifests_estimable": len(finite),
                "manifest_equal_macro": float(np.mean(finite)) if len(finite) else np.nan,
                "manifest_bootstrap_ci95_low": low,
                "manifest_bootstrap_ci95_high": high,
                "bootstrap_valid": valid,
                "interval_type": (
                    "manifest_resampling_descriptive_interval_non_iid_design"
                ),
            }
        )

    formal_route = routing.loc[routing["is_formal"]].copy()
    metric_columns = (
        "top_error_enrichment",
        "top_total_error_capture",
        "reject_remaining_error_reduction",
        "normalized_aurc_50_100",
        "oracle_normalized_utility",
    )
    for keys, group in formal_route.groupby(
        [
            "scope",
            "reference_set",
            "score_name",
            "target_error",
            "outcome_contract",
            "budget",
        ],
        sort=True,
        observed=True,
    ):
        scope, reference_set, score_name, target, contract, budget = keys
        for metric in metric_columns:
            if (
                metric == "normalized_aurc_50_100"
                and abs(float(budget) - 0.20) > 1e-12
            ):
                continue
            values = group[metric].to_numpy(float)
            low, high, valid = bootstrap_mean(
                values,
                stable_int(BOOTSTRAP_SEED, "macro", metric, *keys) % (2**32),
                n_bootstrap,
            )
            finite = values[np.isfinite(values)]
            rows.append(
                {
                    "scope": scope,
                    "reference_set": reference_set,
                    "score_name": score_name,
                    "target_error": target,
                    "outcome_contract": contract,
                    "metric": (
                        metric
                        if metric == "normalized_aurc_50_100"
                        else f"{metric}@{float(budget):.2f}"
                    ),
                    "n_manifests_total": group["manifest_id"].nunique(),
                    "n_manifests_estimable": len(finite),
                    "manifest_equal_macro": float(np.mean(finite)) if len(finite) else np.nan,
                    "manifest_bootstrap_ci95_low": low,
                    "manifest_bootstrap_ci95_high": high,
                    "bootstrap_valid": valid,
                    "interval_type": (
                        "manifest_resampling_descriptive_interval_non_iid_design"
                    ),
                }
            )
    return pd.DataFrame(rows)


def paired_manifest_deltas(
    association: pd.DataFrame,
    routing: pd.DataFrame,
    n_bootstrap: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    assoc = association.loc[
        association["is_formal"]
        & association["score_name"].isin(DISTANCE_SCORES + ("predicted_magnitude_cpa",))
        & association["target_error"].eq("error_cpa_rmse")
    ]
    for (scope, reference_set), group in assoc.groupby(
        ["scope", "reference_set"], sort=True, observed=True
    ):
        pivot = group.pivot_table(
            index="manifest_id",
            columns="score_name",
            values="spearman",
            aggfunc="first",
        )
        for distance_score in DISTANCE_SCORES:
            if distance_score not in pivot or "predicted_magnitude_cpa" not in pivot:
                continue
            matched = pivot[[distance_score, "predicted_magnitude_cpa"]].dropna()
            for manifest_id, item in matched.iterrows():
                rows.append(
                    {
                        "row_type": "manifest",
                        "scope": scope,
                        "reference_set": reference_set,
                        "score_name": distance_score,
                        "comparator": "predicted_magnitude_cpa",
                        "metric": "spearman",
                        "favorable_delta_definition": "distance_minus_magnitude",
                        "manifest_id": manifest_id,
                        "n_manifests": 1,
                        "point_delta": float(
                            item[distance_score] - item["predicted_magnitude_cpa"]
                        ),
                        "ci95_low": np.nan,
                        "ci95_high": np.nan,
                        "bootstrap_unit": "",
                        "bootstrap_replicates": 0,
                        "interval_type": "",
                    }
                )
            values = (
                matched[distance_score] - matched["predicted_magnitude_cpa"]
            ).to_numpy(float)
            low, high, valid = bootstrap_mean(
                values,
                stable_int(
                    BOOTSTRAP_SEED, "paired", scope, reference_set, distance_score, "spearman"
                )
                % (2**32),
                n_bootstrap,
            )
            rows.append(
                {
                    "row_type": "manifest_macro",
                    "scope": scope,
                    "reference_set": reference_set,
                    "score_name": distance_score,
                    "comparator": "predicted_magnitude_cpa",
                    "metric": "spearman",
                    "favorable_delta_definition": "distance_minus_magnitude",
                    "manifest_id": "",
                    "n_manifests": len(values),
                    "point_delta": float(np.mean(values)) if len(values) else np.nan,
                    "ci95_low": low,
                    "ci95_high": high,
                    "bootstrap_unit": "manifest",
                    "bootstrap_replicates": valid,
                    "interval_type": (
                        "manifest_resampling_descriptive_interval_non_iid_design"
                    ),
                }
            )

    route = routing.loc[
        routing["is_formal"]
        & routing["score_name"].isin(DISTANCE_SCORES + ("predicted_magnitude_cpa",))
        & routing["target_error"].eq("error_cpa_rmse")
    ]
    metric_directions = {
        "top_error_enrichment": "distance_minus_magnitude",
        "top_total_error_capture": "distance_minus_magnitude",
        "reject_remaining_error_reduction": "distance_minus_magnitude",
        "normalized_aurc_50_100": "magnitude_minus_distance",
        "oracle_normalized_utility": "distance_minus_magnitude",
    }
    for (scope, reference_set, budget), group in route.groupby(
        ["scope", "reference_set", "budget"], sort=True, observed=True
    ):
        for metric, definition in metric_directions.items():
            if (
                metric == "normalized_aurc_50_100"
                and abs(float(budget) - 0.20) > 1e-12
            ):
                continue
            pivot = group.pivot_table(
                index="manifest_id",
                columns="score_name",
                values=metric,
                aggfunc="first",
            )
            for distance_score in DISTANCE_SCORES:
                if distance_score not in pivot or "predicted_magnitude_cpa" not in pivot:
                    continue
                matched = pivot[[distance_score, "predicted_magnitude_cpa"]].dropna()
                if definition == "magnitude_minus_distance":
                    values = (
                        matched["predicted_magnitude_cpa"] - matched[distance_score]
                    ).to_numpy(float)
                else:
                    values = (
                        matched[distance_score] - matched["predicted_magnitude_cpa"]
                    ).to_numpy(float)
                low, high, valid = bootstrap_mean(
                    values,
                    stable_int(
                        BOOTSTRAP_SEED,
                        "paired",
                        scope,
                        reference_set,
                        budget,
                        distance_score,
                        metric,
                    )
                    % (2**32),
                    n_bootstrap,
                )
                rows.append(
                    {
                        "row_type": "manifest_macro",
                        "scope": scope,
                        "reference_set": reference_set,
                        "score_name": distance_score,
                        "comparator": "predicted_magnitude_cpa",
                        "metric": (
                            metric
                            if metric == "normalized_aurc_50_100"
                            else f"{metric}@{float(budget):.2f}"
                        ),
                        "favorable_delta_definition": definition,
                        "manifest_id": "",
                        "n_manifests": len(values),
                        "point_delta": float(np.mean(values)) if len(values) else np.nan,
                        "ci95_low": low,
                        "ci95_high": high,
                        "bootstrap_unit": "manifest",
                        "bootstrap_replicates": valid,
                        "interval_type": (
                            "manifest_resampling_descriptive_interval_non_iid_design"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def cluster_bootstrap_paired_spearman(
    task_distances: pd.DataFrame,
    n_bootstrap: int,
) -> pd.DataFrame:
    """Resample task keys while preserving the manifest-equal macro estimand."""
    rows: List[Dict[str, Any]] = []
    formal = task_distances.loc[task_distances["is_formal"]].copy()
    for reference_set, group in formal.groupby("reference_set", sort=True, observed=True):
        group = group.reset_index(drop=True)
        task_keys = np.asarray(sorted(group["task_key"].astype(str).unique()))
        manifest_payloads: Dict[str, Dict[str, Any]] = {}
        for manifest_id, manifest in group.groupby(
            "manifest_id", sort=True, observed=True
        ):
            manifest = manifest.reset_index(drop=True)
            manifest_payloads[str(manifest_id)] = {
                "clusters": {
                    str(task): np.asarray(indices, dtype=int)
                    for task, indices in manifest.groupby(
                        "task_key", sort=True
                    ).indices.items()
                },
                "error": manifest["error_cpa_rmse"].to_numpy(float),
                "magnitude": manifest["predicted_magnitude_cpa"].to_numpy(float),
                "distance": {
                    score: manifest[score].to_numpy(float)
                    for score in DISTANCE_SCORES
                },
            }
        rng = np.random.default_rng(
            stable_int(BOOTSTRAP_SEED, "task_cluster", reference_set) % (2**32)
        )
        sampled_sets = rng.integers(
            0, len(task_keys), size=(n_bootstrap, len(task_keys))
        )
        for distance_score in DISTANCE_SCORES:
            point_by_manifest: Dict[str, float] = {}
            for manifest_id, payload in manifest_payloads.items():
                distance_rho = spearman(
                    payload["distance"][distance_score], payload["error"]
                )
                magnitude_rho = spearman(payload["magnitude"], payload["error"])
                delta = distance_rho - magnitude_rho
                if math.isfinite(delta):
                    point_by_manifest[manifest_id] = delta
            required_manifests = tuple(sorted(point_by_manifest))
            if not required_manifests:
                continue
            point_delta = float(np.mean(list(point_by_manifest.values())))
            bootstrap = np.empty(n_bootstrap, dtype=float)
            valid = 0
            for draw in sampled_sets:
                sampled_keys = task_keys[draw]
                manifest_deltas = []
                for manifest_id in required_manifests:
                    payload = manifest_payloads[manifest_id]
                    take_parts = [
                        payload["clusters"][str(task)]
                        for task in sampled_keys
                        if str(task) in payload["clusters"]
                    ]
                    if not take_parts:
                        break
                    take = np.concatenate(take_parts)
                    delta = spearman(
                        payload["distance"][distance_score][take],
                        payload["error"][take],
                    ) - spearman(payload["magnitude"][take], payload["error"][take])
                    if not math.isfinite(delta):
                        break
                    manifest_deltas.append(delta)
                if len(manifest_deltas) == len(required_manifests):
                    bootstrap[valid] = float(np.mean(manifest_deltas))
                    valid += 1
            if valid < int(0.95 * n_bootstrap):
                raise AuditFailure(
                    f"{reference_set}/{distance_score}: too few valid task-cluster draws ({valid})"
                )
            values = bootstrap[:valid]
            low, high = np.quantile(values, [0.025, 0.975])
            rows.append(
                {
                    "row_type": "cluster_sensitivity",
                    "scope": "overall",
                    "reference_set": reference_set,
                    "score_name": distance_score,
                    "comparator": "predicted_magnitude_cpa",
                    "metric": "spearman",
                    "favorable_delta_definition": "distance_minus_magnitude",
                    "manifest_id": "",
                    "n_manifests": len(required_manifests),
                    "point_delta": point_delta,
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                    "bootstrap_unit": (
                        "biological_task_key_cluster_within_manifest_equal_macro"
                    ),
                    "bootstrap_replicates": valid,
                    "n_unique_task_clusters": len(task_keys),
                    "n_repeated_manifest_rows": len(group),
                    "estimand": "manifest_equal_macro_paired_spearman_delta",
                    "interval_type": (
                        "task_key_cluster_descriptive_interval_shared_task_design"
                    ),
                }
            )
    return pd.DataFrame(rows)


def make_invariant_rows(
    spec: Mapping[str, Any],
    loaded_state_hash: str,
    final_state_hash: str,
    prediction_difference: float,
    prediction_count: int,
    task_rows: pd.DataFrame,
    reference_rows: pd.DataFrame,
    reference_audit: Mapping[str, Any],
    train_tasks: Sequence[str],
    validation_tasks: Sequence[str],
    model_contract: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    manifest_id = str(spec["manifest_id"])
    expected_tasks = int(spec["expected_tasks"] or task_rows["task_key"].nunique())
    checks = [
        (
            "strict_state_file_sha256",
            sha256_file(required_manifest_paths(spec)["state"]),
            str(spec["expected_state_sha256"]),
            sha256_file(required_manifest_paths(spec)["state"])
            == str(spec["expected_state_sha256"]),
        ),
        ("strict_load_missing_keys", 0, 0, True),
        ("strict_load_unexpected_keys", 0, 0, True),
        ("module_eval_mode", True, True, True),
        (
            "parameter_hash_unchanged_after_all_inference",
            final_state_hash,
            loaded_state_hash,
            final_state_hash == loaded_state_hash,
        ),
        (
            "pseudo_test_effect_max_abs_difference",
            prediction_difference,
            f"<= {PREDICTION_TOLERANCE}",
            prediction_difference <= PREDICTION_TOLERANCE,
        ),
        (
            "pseudo_test_effect_count",
            prediction_count,
            expected_tasks,
            prediction_count == expected_tasks,
        ),
        (
            "test_task_join_one_to_one",
            task_rows["task_key"].nunique(),
            expected_tasks,
            task_rows["task_key"].nunique() == expected_tasks,
        ),
        (
            "distance_rows_per_task",
            len(task_rows),
            expected_tasks * len(REFERENCE_SETS),
            len(task_rows) == expected_tasks * len(REFERENCE_SETS),
        ),
        (
            "validation_or_test_conditions_in_primary_reference",
            int(reference_audit["reference_validation_overlap_conditions"]),
            0,
            int(reference_audit["reference_validation_overlap_conditions"]) == 0,
        ),
        (
            "all_explicit_train_conditions_preserved",
            int(reference_audit["n_reference_all"]),
            int(reference_audit["n_explicit_train_conditions"]),
            int(reference_audit["n_reference_all"])
            == int(reference_audit["n_explicit_train_conditions"]),
        ),
        (
            "reference_provenance_only_train",
            bool(reference_rows["reference_provenance"].eq("split_cpa=train").all()),
            True,
            bool(reference_rows["reference_provenance"].eq("split_cpa=train").all()),
        ),
        (
            "outcome_columns_absent_from_preoutcome_distance_stage",
            not bool(
                {
                    "error_cpa_rmse",
                    "pair_mean_rmse",
                    "cpa_ridge_disagreement_rmse",
                }
                & set(task_rows.columns)
            ),
            True,
            not bool(
                {
                    "error_cpa_rmse",
                    "pair_mean_rmse",
                    "cpa_ridge_disagreement_rmse",
                }
                & set(task_rows.columns)
            ),
        ),
        (
            "frozen_model_seed",
            int(spec["seed"]),
            int(model_contract["seed"]),
            int(spec["seed"]) == int(model_contract["seed"]),
        ),
        (
            "frozen_train_task_set_sha256",
            sha256_strings(train_tasks),
            str(model_contract["train_task_sha256"]),
            sha256_strings(train_tasks) == str(model_contract["train_task_sha256"]),
        ),
        (
            "frozen_validation_task_set_sha256",
            sha256_strings(validation_tasks),
            str(model_contract["validation_task_sha256"]),
            sha256_strings(validation_tasks)
            == str(model_contract["validation_task_sha256"]),
        ),
        (
            "frozen_primary_reference_condition_sha256",
            sha256_strings(
                reference_rows.loc[
                    reference_rows["reference_set"].eq(
                        "all_explicit_train_conditions"
                    ),
                    "reference_condition",
                ]
            ),
            str(model_contract["primary_reference_condition_sha256"]),
            sha256_strings(
                reference_rows.loc[
                    reference_rows["reference_set"].eq(
                        "all_explicit_train_conditions"
                    ),
                    "reference_condition",
                ]
            )
            == str(model_contract["primary_reference_condition_sha256"]),
        ),
    ]
    return [
        {
            "manifest_id": manifest_id,
            "evidence_role": spec["evidence_role"],
            "gate": gate,
            "observed": observed,
            "required": required,
            "passed": bool(passed),
        }
        for gate, observed, required, passed in checks
    ]


def synthetic_smoke() -> Dict[str, Any]:
    """Exercise distance, routing, constant-score, and clustered resampling helpers."""
    rng = np.random.default_rng(196)
    references = rng.normal(size=(7, 32))
    queries = rng.normal(size=(12, 32))
    cosine = cosine_distances(queries, references)
    euclidean = euclidean_distances(queries, references)
    if cosine.shape != (12, 7) or euclidean.shape != (12, 7):
        raise AuditFailure("synthetic distance smoke failed")
    if not np.isfinite(cosine).all() or not np.isfinite(euclidean).all():
        raise AuditFailure("synthetic distances are nonfinite")

    frame = pd.DataFrame(
        {
            "task_key": [f"task_{index:02d}" for index in range(12)],
            "score": np.arange(12, dtype=float),
            "constant": np.ones(12),
            "error": np.arange(12, dtype=float) + 1.0,
        }
    )
    value = spearman(frame["score"], frame["error"])
    constant = spearman(frame["constant"], frame["error"])
    metrics, curves = routing_for_group(frame, "score", "error", "score")
    if abs(value - 1.0) > 1e-12 or not math.isnan(constant):
        raise AuditFailure("synthetic estimability smoke failed")
    if len(metrics) != len(BUDGETS) or len(curves) != len(COVERAGES):
        raise AuditFailure("synthetic routing smoke failed")
    return {
        "status": "PASS",
        "cosine_shape": list(cosine.shape),
        "euclidean_shape": list(euclidean.shape),
        "monotone_spearman": value,
        "constant_score": "NON_ESTIMABLE",
        "routing_budget_rows": len(metrics),
        "risk_coverage_rows": len(curves),
    }


def plot_results(macro: pd.DataFrame, paired: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary_reference = "all_explicit_train_conditions"
    overall = macro.loc[
        macro["scope"].eq("overall")
        & macro["reference_set"].eq(primary_reference)
        & macro["target_error"].eq("error_cpa_rmse")
    ].copy()
    labels = {
        "native_cosine_distance": "CPA cosine distance",
        "native_euclidean_distance": "CPA Euclidean distance",
        "predicted_magnitude_cpa": "CPA predicted magnitude",
    }
    colors = {
        "native_cosine_distance": "#35608D",
        "native_euclidean_distance": "#5B8E7D",
        "predicted_magnitude_cpa": "#A35D3A",
    }
    order = list(labels)

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4), facecolor="white")
    panels = [
        ("spearman", "Spearman with CPA RMSE", True),
        ("normalized_aurc_50_100", "Normalized AURC (lower is better)", False),
        ("oracle_normalized_utility@0.20", "20% oracle-normalized utility", True),
    ]
    for ax, (metric, title, _) in zip(axes.ravel()[:3], panels):
        subset = overall.loc[overall["metric"].eq(metric)].set_index("score_name")
        available = [score for score in order if score in subset.index]
        values = [subset.loc[score, "manifest_equal_macro"] for score in available]
        low = [subset.loc[score, "manifest_bootstrap_ci95_low"] for score in available]
        high = [subset.loc[score, "manifest_bootstrap_ci95_high"] for score in available]
        positions = np.arange(len(available))
        ax.bar(
            positions,
            values,
            color=[colors[score] for score in available],
            width=0.68,
            edgecolor="white",
        )
        ax.errorbar(
            positions,
            values,
            yerr=[
                np.asarray(values) - np.asarray(low),
                np.asarray(high) - np.asarray(values),
            ],
            fmt="none",
            ecolor="#263238",
            capsize=3,
            lw=1,
        )
        ax.set_xticks(positions)
        ax.set_xticklabels([labels[score] for score in available], rotation=17, ha="right")
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.grid(axis="y", color="#E6E9EC", lw=0.8)
        ax.set_axisbelow(True)
        ax.set_facecolor("white")

    ax = axes.ravel()[3]
    delta = paired.loc[
        paired["row_type"].eq("manifest_macro")
        & paired["scope"].eq("overall")
        & paired["reference_set"].eq(primary_reference)
        & paired["metric"].eq("spearman")
    ].set_index("score_name")
    available = [score for score in DISTANCE_SCORES if score in delta.index]
    values = [delta.loc[score, "point_delta"] for score in available]
    low = [delta.loc[score, "ci95_low"] for score in available]
    high = [delta.loc[score, "ci95_high"] for score in available]
    positions = np.arange(len(available))
    ax.bar(
        positions,
        values,
        color=[colors[score] for score in available],
        width=0.58,
        edgecolor="white",
    )
    ax.errorbar(
        positions,
        values,
        yerr=[
            np.asarray(values) - np.asarray(low),
            np.asarray(high) - np.asarray(values),
        ],
        fmt="none",
        ecolor="#263238",
        capsize=3,
        lw=1,
    )
    ax.axhline(0.0, color="#6F777C", lw=1, ls="--")
    ax.set_xticks(positions)
    ax.set_xticklabels([labels[score] for score in available], rotation=17, ha="right")
    ax.set_title("ΔSpearman vs magnitude", loc="left", fontsize=11, fontweight="bold")
    ax.grid(axis="y", color="#E6E9EC", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_facecolor("white")

    fig.suptitle(
        "E196  |  CPA native latent support distance",
        x=0.06,
        y=0.995,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.955,
        (
            "Eight frozen sciPlex3 manifests; error bars are descriptive "
            "10,000-draw manifest-resampling intervals."
        ),
        ha="left",
        fontsize=9,
        color="#56636B",
    )
    fig.tight_layout(rect=[0.04, 0.03, 0.99, 0.93])
    for suffix in ("png", "pdf"):
        fig.savefig(
            FIGURES / f"E196_cpa_native_support_distance.{suffix}",
            dpi=300 if suffix == "png" else None,
            facecolor="white",
            bbox_inches="tight",
        )
    plt.close(fig)


def write_reports(
    task_distances: pd.DataFrame,
    macro: pd.DataFrame,
    paired: pd.DataFrame,
    invariants: pd.DataFrame,
    reference_audits: pd.DataFrame,
    environment: pd.DataFrame,
    command: str,
    started: str,
    elapsed_seconds: float,
    smoke: Mapping[str, Any],
) -> None:
    primary = macro.loc[
        macro["scope"].eq("overall")
        & macro["reference_set"].eq("all_explicit_train_conditions")
        & macro["target_error"].eq("error_cpa_rmse")
        & macro["metric"].isin(
            ["spearman", "normalized_aurc_50_100", "oracle_normalized_utility@0.20"]
        )
    ][
        [
            "score_name",
            "metric",
            "n_manifests_estimable",
            "manifest_equal_macro",
            "manifest_bootstrap_ci95_low",
            "manifest_bootstrap_ci95_high",
        ]
    ].copy()
    primary["score_name"] = primary["score_name"].replace(
        {
            "native_cosine_distance": "CPA cosine support distance",
            "native_euclidean_distance": "CPA Euclidean support distance",
            "predicted_magnitude_cpa": "CPA predicted magnitude",
        }
    )
    manifest_deltas = paired.loc[
        paired["row_type"].eq("manifest_macro")
        & paired["scope"].eq("overall")
        & paired["reference_set"].eq("all_explicit_train_conditions")
        & paired["metric"].eq("spearman")
    ][
        ["score_name", "point_delta", "ci95_low", "ci95_high", "n_manifests"]
    ].rename(
        columns={
            "ci95_low": "manifest_interval_low",
            "ci95_high": "manifest_interval_high",
            "n_manifests": "n_manifests_estimable",
        }
    )
    cluster_deltas = paired.loc[
        paired["row_type"].eq("cluster_sensitivity")
        & paired["scope"].eq("overall")
        & paired["reference_set"].eq("all_explicit_train_conditions")
        & paired["metric"].eq("spearman")
    ][
        [
            "score_name",
            "ci95_low",
            "ci95_high",
            "n_unique_task_clusters",
            "point_matches_manifest_macro",
        ]
    ].rename(
        columns={
            "ci95_low": "task_cluster_interval_low",
            "ci95_high": "task_cluster_interval_high",
        }
    )
    deltas = manifest_deltas.merge(
        cluster_deltas,
        on="score_name",
        how="left",
        validate="one_to_one",
    )
    control_summary = reference_audits[
        [
            "manifest_id",
            "n_reference_all",
            "n_reference_gt30",
            "n_reference_perturbed",
            "n_control_nearest_cosine_all",
            "n_control_nearest_euclidean_all",
        ]
    ]
    failed = invariants.loc[~invariants["passed"].astype(bool)]
    report = f"""# E196｜CPA 原生潜空间支持距离审计

E196 严格加载 E94 开发适配器和 E84 八个 formal manifest 的已有 CPA 权重，
不重训模型，也不修改任何 reference set。这里的 CPA 原生分数是目标
`cell line + dose-weighted drug` 潜向量到训练条件潜向量的最近距离，不是预测方差、
校准概率或误差下界。

- 证据标签：`{EVIDENCE_LABEL}`
- formal manifest：{len(FORMAL_MANIFESTS)}
- formal manifest-task：{task_distances.loc[task_distances['is_formal'], ['manifest_id', 'task_key']].drop_duplicates().shape[0]}
- 开发适配器任务：{task_distances.loc[~task_distances['is_formal'], 'task_key'].nunique()}
- reference set：无阈值显式 train、官方 `>30` 敏感性、perturbed-train-only
- invariant failures：{len(failed)}

## 同一 CPA outcome 的主结果

{markdown_table(primary)}

## support distance 相对 magnitude 的配对 ΔSpearman

正值表示 support distance 的相关更高。表中并列给出 manifest-resampling
描述性区间，以及保持同一“manifest 内计算、manifest 间等权”估计量的
biological-task cluster 描述性区间。八个 manifest 共享任务且不是 iid 抽样，
10,000 只是 Monte Carlo 重采样次数，不增加有效样本量。

{markdown_table(deltas)}

## 参考条件与 control 最近邻

{markdown_table(control_summary, digits=0)}

E94 只做模型重建、任务连接和预测复现检查，不进入 formal 宏平均。所有四个
Cartesian quadrant、负结果、constant-score 标记和 reference-set 敏感性均保留。
CPA–ridge disagreement 对 pair-mean RMSE 作为不同 predictor family 的补充，
没有与 CPA 自身 RMSE 的 head-to-head 结果混写。
"""
    (REPORTS / "E196_REPORT.md").write_text(report)

    primary_delta = deltas.sort_values("score_name")
    if primary_delta.empty:
        interpretation_text = "主参考集合没有可估计的配对结果。"
    else:
        statements = []
        for row in primary_delta.itertuples(index=False):
            if abs(float(row.point_delta)) <= 1e-12:
                direction = "与 magnitude 持平"
            elif row.point_delta > 0:
                direction = "数值高于 magnitude"
            else:
                direction = "数值低于 magnitude"
            manifest_positive = row.manifest_interval_low > 0
            manifest_negative = row.manifest_interval_high < 0
            cluster_positive = row.task_cluster_interval_low > 0
            cluster_negative = row.task_cluster_interval_high < 0
            if manifest_positive and cluster_positive:
                uncertainty = "两种描述性区间均在 0 以上"
            elif manifest_negative and cluster_negative:
                uncertainty = "两种描述性区间均在 0 以下"
            else:
                uncertainty = "两种依赖处理未共同给出单一方向"
            estimable_note = (
                "8 个 manifest 均可估"
                if int(row.n_manifests_estimable) == len(FORMAL_MANIFESTS)
                else f"仅 {int(row.n_manifests_estimable)} 个 manifest 可估"
            )
            statements.append(
                f"- `{row.score_name}` 的宏平均 Δρ={row.point_delta:.3f}，"
                f"{direction}；manifest 描述性区间 "
                f"[{row.manifest_interval_low:.3f}, {row.manifest_interval_high:.3f}]，"
                f"task-cluster 描述性区间 "
                f"[{row.task_cluster_interval_low:.3f}, "
                f"{row.task_cluster_interval_high:.3f}]；"
                f"{uncertainty}，{estimable_note}。"
            )
        interpretation_text = "\n".join(statements)
    interpretation = f"""# E196｜解释边界

## 可以写

- 这是 CPA 0.8.8 自带语义下的训练支持距离审计。
- distance 与 magnitude 都对同一 CPA predictor 的 RMSE 评价，属于可比的
  same-outcome audit。
- frozen manifest 先各自计算，再等权宏平均；task-key cluster 重采样保持相同
  估计量，只作共享生物任务依赖的描述性敏感性。

## 结果如何读

{interpretation_text}

## 不能写

- 不能称作 predictive variance、校准置信度、误差概率或误差下界。
- 不能把 E94 或八个 manifest 写成九个独立外部数据集。
- 不能按观察到的 RMSE 改 reference set、距离、阈值或任务。
- 不能把 CPA–ridge disagreement 的 pair-mean target 与 CPA distance 的 own-RMSE
  target 合成同一个优劣结论。
- 不能把重采样描述性区间当作八个独立外部样本形成的常规置信区间。
"""
    (REPORTS / "E196_INTERPRETATION.md").write_text(interpretation)

    environment_map = environment.set_index("key")["value"].to_dict()
    run_record = f"""# E196｜运行记录

- started：`{started}`
- finished：`{utcish_now()}`
- elapsed seconds：`{elapsed_seconds:.3f}`
- command：`{command}`
- executable：`{environment_map.get('python_executable')}`
- Python：`{environment_map.get('python_version')}`
- CPA：`{environment_map.get('package::cpa-tools')}`
- torch：`{environment_map.get('package::torch')}`
- CPA source commit：`{environment_map.get('cpa_source_commit')}`
- CPA worktree：`{environment_map.get('cpa_source_worktree_status') or 'clean'}`
- CPA runtime source：`__init__.py` 的 Ray 可选导入兼容补丁与
  `_api.py/_model.py/_module.py` 均由 `E196_CODE_LOCK.json` 锁定；核心模型与
  uncertainty 实现文件没有未记录漂移
- model retraining：`false`
- covariate embedding：按冻结 `covars_encoder` 类别编号直接读取同一
  `module.covars_embeddings`；规避 CPA 0.8.8 scalar wrapper 的三维 AnnData 兼容缺陷
- synthetic smoke：`{smoke.get('status')}`
- pseudo-test reproduction tolerance：`{PREDICTION_TOLERANCE}`
- manifest/task-key descriptive resampling：`{N_BOOTSTRAP}`
- output hash index：`tables/E196_INPUT_HASHES.csv`

冻结 CPA builder 会把来源 h5ad 载入进程，用于重建训练输入、pseudo-control
预测复现和 control mean；target perturbed expression 与 error 列不进入距离函数。
全部九个模型的距离阶段完成并写入带哈希的 pre-outcome 文件后，程序才解析并连接
既有 RMSE。运行失败时 `E196_STATUS.json` 写为 `FAILED`，不得沿用部分表得出结论。
"""
    (REPORTS / "RUN_RECORD.md").write_text(run_record)


def write_output_hashes(input_hashes: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = list(input_hashes)
    candidates = [OUT / "E196_STATUS.json"]
    for directory in (TABLES, FIGURES, REPORTS):
        candidates.extend(path for path in directory.rglob("*") if path.is_file())
    excluded = {
        TABLES / "E196_INPUT_HASHES.csv",
    }
    for path in sorted(set(candidates) - excluded):
        rows.append(
            {
                "artifact_scope": "output",
                "artifact_role": "generated_e196_artifact",
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "expected_sha256": "",
                "observed_sha256": sha256_file(path),
                "hash_locked": False,
                "hash_match": "",
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(TABLES / "E196_INPUT_HASHES.csv", index=False)
    return frame


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def write_running_status(mode: str, started: str) -> None:
    atomic_write_json(
        OUT / "E196_STATUS.json",
        {
            "experiment": "E196",
            "stage": EVIDENCE_LABEL,
            "status": "RUNNING",
            "mode": mode,
            "started_at": started,
            "model_retraining": False,
            "claim_boundary": (
                "No E196 scientific conclusion is authorized until status is COMPLETE."
            ),
        },
    )


def write_failed_status(mode: str, started: str, exc: BaseException) -> None:
    payload = {
        "experiment": "E196",
        "stage": EVIDENCE_LABEL,
        "status": "FAILED",
        "mode": mode,
        "started_at": started,
        "failed_at": utcish_now(),
        "model_retraining": False,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "claim_boundary": (
            "No E196 scientific conclusion is authorized from a failed or partial run."
        ),
    }
    atomic_write_json(OUT / "E196_STATUS.json", payload)


def run_analysis(
    args: argparse.Namespace,
    input_hashes: List[Dict[str, Any]],
    environment: pd.DataFrame,
    smoke: Mapping[str, Any],
    command: str,
    started: str,
) -> Dict[str, Any]:
    start_time = time.monotonic()
    for directory in (OUT, TABLES, FIGURES, REPORTS):
        directory.mkdir(parents=True, exist_ok=True)

    core = load_e83_core()
    code_lock = json.loads(CODE_LOCK.read_text())
    model_contracts = code_lock.get("model_contracts")
    if not isinstance(model_contracts, dict):
        raise AuditFailure("E196 code lock lacks model_contracts")
    all_tasks: List[pd.DataFrame] = []
    all_references: List[pd.DataFrame] = []
    reference_audits: List[Dict[str, Any]] = []
    invariant_rows: List[Dict[str, Any]] = []
    preoutcome_by_manifest: Dict[str, pd.DataFrame] = {}

    for spec in manifest_specs():
        manifest_id = str(spec["manifest_id"])
        if manifest_id not in model_contracts:
            raise AuditFailure(f"{manifest_id}: frozen model contract is missing")
        print(f"[E196] rebuilding {manifest_id}", flush=True)
        (
            model,
            combined,
            manifest,
            source,
            x,
            obs,
            loaded_state_hash,
            train_tasks,
            validation_tasks,
        ) = build_and_load_model(core, spec, args.device)
        prediction_difference, prediction_count = reproduce_predictions(
            model, combined, manifest, x, obs, spec
        )
        truth_free_tasks, references, reference_audit = extract_distances(
            model, combined, manifest, spec
        )
        final_state_hash = state_digest(model.module.state_dict())
        reference_audit.update(
            {
                "evidence_role": spec["evidence_role"],
                "n_train_tasks_after_validation_carveout": len(train_tasks),
                "n_validation_tasks": len(validation_tasks),
                "pseudo_test_effect_max_abs_difference": prediction_difference,
                "loaded_parameter_hash": loaded_state_hash,
                "final_parameter_hash": final_state_hash,
            }
        )
        preoutcome_by_manifest[manifest_id] = truth_free_tasks
        all_references.append(references)
        reference_audits.append(reference_audit)
        invariant_rows.extend(
            make_invariant_rows(
                spec,
                loaded_state_hash,
                final_state_hash,
                prediction_difference,
                prediction_count,
                truth_free_tasks,
                references,
                reference_audit,
                train_tasks,
                validation_tasks,
                model_contracts[manifest_id],
            )
        )
        del model, combined, manifest, source, x, obs
        gc.collect()

    preoutcome_tasks = pd.concat(
        [preoutcome_by_manifest[str(spec["manifest_id"])] for spec in manifest_specs()],
        ignore_index=True,
    )
    reference_conditions = pd.concat(all_references, ignore_index=True)
    reference_audit_frame = pd.DataFrame(reference_audits)
    invariants = pd.DataFrame(invariant_rows)
    if not invariants["passed"].astype(bool).all():
        failed = invariants.loc[~invariants["passed"].astype(bool)]
        raise AuditFailure(
            "implementation gates failed:\n" + failed.to_string(index=False)
        )

    # Persist and hash the complete distance stage before any outcome CSV is
    # parsed.  The source h5ad is necessarily loaded by the frozen CPA builder,
    # but target expression and error columns do not enter distance functions.
    preoutcome_task_path = TABLES / "E196_PREOUTCOME_TASK_DISTANCES.csv"
    preoutcome_reference_path = TABLES / "E196_PREOUTCOME_REFERENCE_CONDITIONS.csv"
    preoutcome_tasks.to_csv(preoutcome_task_path, index=False)
    reference_conditions.to_csv(preoutcome_reference_path, index=False)
    atomic_write_json(
        TABLES / "E196_PREOUTCOME_PROVENANCE.json",
        {
            "stage": "PREOUTCOME_DISTANCE_COMPLETE",
            "completed_at": utcish_now(),
            "task_rows": len(preoutcome_tasks),
            "reference_rows": len(reference_conditions),
            "task_csv_sha256": sha256_file(preoutcome_task_path),
            "reference_csv_sha256": sha256_file(preoutcome_reference_path),
            "outcome_columns_present": sorted(
                {
                    "error_cpa_rmse",
                    "pair_mean_rmse",
                    "cpa_ridge_disagreement_rmse",
                }
                & set(preoutcome_tasks.columns)
            ),
            "truth_boundary": (
                "The frozen CPA builder loads the source h5ad in-process; target "
                "perturbed expression and error columns are not arguments to or "
                "numerical inputs of the distance functions."
            ),
        },
    )

    for spec in manifest_specs():
        manifest_id = str(spec["manifest_id"])
        all_tasks.append(join_outcomes(preoutcome_by_manifest[manifest_id], spec))
    task_distances = pd.concat(all_tasks, ignore_index=True)

    dynamic, association, routing, curves = compute_statistics(task_distances)
    if dynamic["n_nan"].sum() or dynamic["n_infinite"].sum():
        raise AuditFailure("native score dynamic-range audit found NaN/Inf")
    if not dynamic.loc[
        dynamic["estimability"].eq("NON_ESTIMABLE"), "n_unique_rounded_12dp"
    ].le(1).all():
        raise AuditFailure("NON_ESTIMABLE scores were not recorded consistently")

    macro = macro_summary(association, routing, args.bootstrap)
    paired = paired_manifest_deltas(association, routing, args.bootstrap)
    cluster = cluster_bootstrap_paired_spearman(task_distances, args.bootstrap)
    headline = macro.loc[
        macro["scope"].eq("overall")
        & macro["reference_set"].eq("all_explicit_train_conditions")
        & macro["target_error"].eq("error_cpa_rmse")
        & macro["score_name"].isin(SAME_OUTCOME_SCORES)
        & macro["metric"].eq("spearman")
    ]
    if (
        len(headline) != len(SAME_OUTCOME_SCORES)
        or not headline["n_manifests_total"].eq(len(FORMAL_MANIFESTS)).all()
        or not headline["n_manifests_estimable"].eq(len(FORMAL_MANIFESTS)).all()
    ):
        raise AuditFailure(
            "primary overall same-outcome headline is not estimable in all "
            f"{len(FORMAL_MANIFESTS)} formal manifests"
        )
    if (
        len(
            cluster.loc[
                cluster["reference_set"].eq("all_explicit_train_conditions")
                & cluster["score_name"].isin(DISTANCE_SCORES)
            ]
        )
        != len(DISTANCE_SCORES)
        or not cluster.loc[
            cluster["reference_set"].eq("all_explicit_train_conditions")
            & cluster["score_name"].isin(DISTANCE_SCORES),
            "n_manifests",
        ]
        .eq(len(FORMAL_MANIFESTS))
        .all()
    ):
        raise AuditFailure(
            "primary task-key cluster sensitivity is not estimable in all "
            f"{len(FORMAL_MANIFESTS)} formal manifests"
        )
    for index, cluster_row in cluster.iterrows():
        matched = paired.loc[
            paired["row_type"].eq("manifest_macro")
            & paired["scope"].eq(cluster_row["scope"])
            & paired["reference_set"].eq(cluster_row["reference_set"])
            & paired["score_name"].eq(cluster_row["score_name"])
            & paired["metric"].eq(cluster_row["metric"])
        ]
        if len(matched) != 1:
            raise AuditFailure(
                "cluster sensitivity could not match one manifest-macro estimate"
            )
        difference = abs(
            float(cluster_row["point_delta"]) - float(matched["point_delta"].iloc[0])
        )
        if difference > 1e-12:
            raise AuditFailure(
                "task-key cluster point estimate changed the frozen manifest-macro "
                f"estimand ({difference})"
            )
        cluster.loc[index, "point_matches_manifest_macro"] = True
    paired = pd.concat([paired, cluster], ignore_index=True, sort=False)

    expected_task_rows = (629 + 59) * len(REFERENCE_SETS)
    if len(task_distances) != expected_task_rows:
        raise AuditFailure(
            f"expected {expected_task_rows} manifest-task-reference rows, observed {len(task_distances)}"
        )
    if (
        task_distances.loc[task_distances["is_formal"], ["manifest_id", "task_key"]]
        .drop_duplicates()
        .shape[0]
        != 629
    ):
        raise AuditFailure("formal manifest-task count changed")

    task_distances.to_csv(TABLES / "E196_TASK_DISTANCES.csv", index=False)
    reference_conditions.to_csv(TABLES / "E196_REFERENCE_CONDITIONS.csv", index=False)
    dynamic.to_csv(TABLES / "E196_DYNAMIC_RANGE_AUDIT.csv", index=False)
    association.to_csv(TABLES / "E196_ASSOCIATION.csv", index=False)
    paired.to_csv(TABLES / "E196_PAIRED_DELTAS.csv", index=False)
    routing.to_csv(TABLES / "E196_ROUTING_METRICS.csv", index=False)
    curves.to_csv(TABLES / "E196_RISK_COVERAGE.csv", index=False)
    macro.to_csv(TABLES / "E196_QUADRANT_SUMMARY.csv", index=False)
    invariants.to_csv(TABLES / "E196_INVARIANT_AUDIT.csv", index=False)
    environment.to_csv(TABLES / "E196_RUNTIME_ENVIRONMENT.csv", index=False)

    plot_results(macro, paired)
    elapsed = time.monotonic() - start_time
    write_reports(
        task_distances,
        macro,
        paired,
        invariants,
        reference_audit_frame,
        environment,
        command,
        started,
        elapsed,
        smoke,
    )
    output_candidates = []
    for directory in (TABLES, FIGURES, REPORTS):
        output_candidates.extend(
            path for path in directory.rglob("*") if path.is_file()
        )
    output_hash_count = (
        len(
            set(output_candidates)
            - {TABLES / "E196_INPUT_HASHES.csv"}
        )
        + 1
    )

    status = {
        "experiment": "E196",
        "stage": EVIDENCE_LABEL,
        "status": "COMPLETE",
        "mode": args.mode,
        "started_at": started,
        "completed_at": utcish_now(),
        "elapsed_seconds": elapsed,
        "development_manifest": "E81_r1_p75",
        "formal_manifests": list(FORMAL_MANIFESTS),
        "n_formal_manifest_tasks": 629,
        "n_development_tasks": 59,
        "n_task_reference_rows": len(task_distances),
        "n_reference_rows": len(reference_conditions),
        "reference_sets": list(REFERENCE_SETS),
        "model_retraining": False,
        "state_updates": False,
        "all_invariants_passed": True,
        "invariant_count": len(invariants),
        "pseudo_test_effect_tolerance": PREDICTION_TOLERANCE,
        "maximum_pseudo_test_effect_abs_difference": float(
            reference_audit_frame["pseudo_test_effect_max_abs_difference"].max()
        ),
        "manifest_bootstrap_replicates": args.bootstrap,
        "task_key_cluster_bootstrap_replicates": args.bootstrap,
        "output_hash_count": output_hash_count,
        "synthetic_smoke": dict(smoke),
        "primary_outcome": "error_cpa_rmse",
        "primary_reference_set": "all_explicit_train_conditions",
        "claim_boundary": (
            "E196 evaluates association and routing utility for CPA own error; "
            "it is post-truth, descriptive, not predictive variance, and not a "
            "new blind validation."
        ),
    }
    atomic_write_json(OUT / "E196_STATUS.json", status)
    hashes = write_output_hashes(input_hashes)
    observed_output_hash_count = int(hashes["artifact_scope"].eq("output").sum())
    if observed_output_hash_count != output_hash_count:
        raise AuditFailure(
            f"output hash count changed: expected {output_hash_count}, "
            f"observed {observed_output_hash_count}"
        )
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen E196 CPA native latent support-distance audit."
    )
    parser.add_argument(
        "--mode",
        choices=("preflight", "full"),
        default="full",
        help=(
            "preflight verifies locks only; full runs the locked synthetic smoke "
            "and the formal audit."
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="torch inference device; CPU is the reproducibility default.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=N_BOOTSTRAP,
        help="Bootstrap replicates. The frozen formal value is 10000.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = utcish_now()
    command = " ".join([sys.executable] + sys.argv)
    try:
        if args.mode != "preflight":
            write_running_status(args.mode, started)
        if args.bootstrap != N_BOOTSTRAP:
            raise AuditFailure(
                f"formal E196 bootstrap is frozen at {N_BOOTSTRAP}; "
                f"observed {args.bootstrap}"
            )
        smoke: Dict[str, Any] = {"status": "NOT_REQUESTED"}
        if args.mode == "full":
            smoke = synthetic_smoke()
        input_hashes, environment = preflight(command)
        if args.mode == "preflight":
            result = {
                "experiment": "E196",
                "mode": "preflight",
                "status": "PASS",
                "checked_input_artifacts": len(input_hashes),
                "environment": environment.set_index("key")["value"].to_dict(),
            }
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        status = run_analysis(
            args,
            input_hashes,
            environment,
            smoke,
            command,
            started,
        )
        print(json.dumps(status, ensure_ascii=False, indent=2))
    except Exception as exc:
        if args.mode != "preflight":
            write_failed_status(args.mode, started, exc)
        raise


if __name__ == "__main__":
    main()
