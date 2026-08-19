#!/usr/bin/env python3
"""Build the physically isolated expression assets for E168.

This is deliberately a two-stage, fail-closed splitter:

* ``--stage pretruth`` reads only the rows registered as controls, supervised
  training targets, or validation targets.  It never reads test targeting X
  or the permanently forbidden column-unseen train/validation X values.
* ``--stage postgate`` reads only the registered test-targeting rows, and only
  after a committed PASS snapshot is present on both configured remotes.

The 44.6 GB source H5AD is not copied into either isolated directory.  Every
source row returned by HDF5 is recorded exactly once in the corresponding
access audit.  The script computes no test performance metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "E168_primary_human_cd4_fresh_confirmation"
TASK_PREFIX = "E168"
PRETRUTH_ASSET_STAGE = "F2_PRETRUTH_ISOLATED_ASSET_BUILD"
POSTGATE_ASSET_STAGE = "F3_POSTGATE_ISOLATED_TRUTH_BUILD"
PRETRUTH_GATE_STAGE = "F2_PRETRUTH_GATE"
EXPERIMENT_REL = Path(
    "docs/实验结果/E168_primary_human_cd4_fresh_confirmation_20260716"
)
EXPERIMENT = ROOT / EXPERIMENT_REL
FROZEN_METADATA_COMMIT = "83acafb98da67ed65507abb8967dda69e9436de7"

SOURCE_LOCK_REL = EXPERIMENT_REL / "SOURCE_LOCK.json"
RUN_STATUS_REL = EXPERIMENT_REL / "RUN_STATUS.json"
MODEL_LOCK_REL = EXPERIMENT_REL / "MODEL_INPUT_LOCK.json"
STAT_LOCK_REL = EXPERIMENT_REL / "STATISTICAL_ANALYSIS_LOCK.json"
PLAN_REL = EXPERIMENT_REL / "PREREG_ANALYSIS_PLAN.md"
DONOR_ROLES_REL = EXPERIMENT_REL / "manifests/E168_DONOR_STATE_ROLES.csv"
ROW_ACCESS_REL = EXPERIMENT_REL / "manifests/E168_ROW_ACCESS_MANIFEST.csv"
TARGETS_REL = EXPERIMENT_REL / "manifests/E168_SELECTED_TARGETS.csv"
TASKS_REL = EXPERIMENT_REL / "manifests/E168_TASK_MANIFEST.csv"

FROZEN_INPUTS = (
    SOURCE_LOCK_REL,
    RUN_STATUS_REL,
    MODEL_LOCK_REL,
    STAT_LOCK_REL,
    PLAN_REL,
    DONOR_ROLES_REL,
    ROW_ACCESS_REL,
    TARGETS_REL,
    TASKS_REL,
)

DATA_ROOT = Path("/home/yyf/data/safeconf_external/primary_cd4_perturbseq_2025")
BYTE_ATTESTATION = DATA_ROOT / "E168_SOURCE_BYTE_ATTESTATION.json"
ISOLATED_ROOT = DATA_ROOT / "isolated"
F2_DIR = ISOLATED_ROOT / "F2_pretruth"
F3_DIR = ISOLATED_ROOT / "F3_postgate"
EXPECTED_STATES = ("Rest", "Stim8hr", "Stim48hr")
PANEL_SIZE = 512
N_TARGET_GENES = 200
N_EXTRA_GENES = PANEL_SIZE - N_TARGET_GENES
EXPECTED_PHASE_COUNTS = {
    "PRETRUTH_CONTROL_X": 11018,
    "PRETRUTH_TRAIN_X": 1920,
    "PRETRUTH_VALIDATION_X": 960,
    "POSTGATE_TEST_TRUTH_X": 1200,
    "FORBIDDEN_COLUMN_UNSEEN_X": 720,
}
PRETRUTH_PHASES = (
    "PRETRUTH_CONTROL_X",
    "PRETRUTH_TRAIN_X",
    "PRETRUTH_VALIDATION_X",
)
POSTGATE_PHASES = ("POSTGATE_TEST_TRUTH_X",)
BUILDER_REL = Path("tools/scripts/build_e168_primary_cd4_isolated_assets.py")
DOWNLOADER_REL = Path("tools/scripts/download_e168_primary_cd4.py")


class IntegrityError(RuntimeError):
    """A frozen input, source byte, or access boundary did not match."""


def sha256_file(path: Path, chunk_size: int = 64 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=check, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def git_text(*args: str) -> str:
    return git(*args).stdout.decode("utf-8").strip()


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def decode_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype.kind not in "OSU":
        return values
    return np.asarray(
        [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in values],
        dtype=object,
    )


def read_categorical_or_array(handle: h5py.File, path: str) -> np.ndarray:
    node = handle[path]
    if isinstance(node, h5py.Group):
        if set(node.keys()) != {"categories", "codes"}:
            raise IntegrityError(
                f"Unexpected categorical representation at {path}: {sorted(node.keys())}"
            )
        categories = decode_array(node["categories"][...])
        codes = np.asarray(node["codes"][...], dtype=np.int64)
        if np.any(codes >= len(categories)):
            raise IntegrityError(f"Out-of-range categorical code at {path}")
        return np.asarray(
            [categories[code] if code >= 0 else "" for code in codes], dtype=object
        )
    return decode_array(node[...])


def normalize_merged_raw_counts(raw_counts: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw_counts, dtype=np.float64)
    if raw.ndim != 1:
        raise IntegrityError(f"Expected one count vector, got shape={raw.shape}")
    if not np.all(np.isfinite(raw)) or np.any(raw < 0):
        raise IntegrityError("Raw-count vector contains a non-finite or negative value")
    library_sum = float(raw.sum(dtype=np.float64))
    if not math.isfinite(library_sum) or library_sum <= 0:
        raise IntegrityError(f"Invalid raw library sum: {library_sum}")
    return np.log1p(1.0e4 * raw / library_sum)


def normalize_panel_from_merged_counts(
    panel_counts: np.ndarray, library_sum: float
) -> np.ndarray:
    values = np.asarray(panel_counts, dtype=np.float64)
    if not math.isfinite(library_sum) or library_sum <= 0:
        raise IntegrityError(f"Invalid raw library sum: {library_sum}")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise IntegrityError("Panel raw counts contain a non-finite or negative value")
    return np.log1p(1.0e4 * values / library_sum)


class RowMatrixReader:
    """Read only explicitly requested logical rows from AnnData X."""

    def __init__(self, handle: h5py.File):
        self.node = handle["X"]
        self.encoding: str
        if isinstance(self.node, h5py.Dataset):
            if len(self.node.shape) != 2:
                raise IntegrityError(f"Dense X is not two-dimensional: {self.node.shape}")
            self.encoding = "dense"
            self.shape = tuple(map(int, self.node.shape))
        else:
            raw_encoding = self.node.attrs.get("encoding-type", "")
            if isinstance(raw_encoding, bytes):
                raw_encoding = raw_encoding.decode("utf-8")
            self.encoding = str(raw_encoding).lower()
            if self.encoding not in {"csr_matrix", "csr"}:
                raise IntegrityError(
                    "E168 requires logical row access; unsupported X encoding "
                    f"{self.encoding!r} (CSC would require reading unregistered rows)"
                )
            if not {"data", "indices", "indptr"}.issubset(self.node.keys()):
                raise IntegrityError("CSR X is missing data/indices/indptr")
            shape = self.node.attrs.get("shape")
            if shape is None:
                shape = (len(self.node["indptr"]) - 1, int(self.node["indices"][:].max()) + 1)
            self.shape = tuple(map(int, shape))

    def read_rows(self, row_indices: Sequence[int]) -> np.ndarray:
        rows = np.asarray(row_indices, dtype=np.int64)
        if rows.ndim != 1 or len(rows) == 0:
            raise IntegrityError("read_rows requires a non-empty one-dimensional index")
        if np.any(rows < 0) or np.any(rows >= self.shape[0]):
            raise IntegrityError("Requested X row is outside the source matrix")
        if len(np.unique(rows)) != len(rows):
            raise IntegrityError("A source X row was requested twice in one batch")
        if np.any(rows[1:] <= rows[:-1]):
            raise IntegrityError("X rows must be requested in strictly increasing order")
        if self.encoding == "dense":
            values = np.asarray(self.node[rows, :], dtype=np.float64)
        else:
            values = np.zeros((len(rows), self.shape[1]), dtype=np.float64)
            indptr = self.node["indptr"]
            data = self.node["data"]
            indices = self.node["indices"]
            for output_i, source_i in enumerate(rows.tolist()):
                bounds = np.asarray(indptr[source_i : source_i + 2], dtype=np.int64)
                start, stop = map(int, bounds)
                cols = np.asarray(indices[start:stop], dtype=np.int64)
                row_values = np.asarray(data[start:stop], dtype=np.float64)
                if len(cols) != len(np.unique(cols)):
                    raise IntegrityError(f"CSR X row {source_i} contains duplicate columns")
                values[output_i, cols] = row_values
        if values.shape != (len(rows), self.shape[1]):
            raise IntegrityError(
                f"Unexpected row-read shape {values.shape}; expected {(len(rows), self.shape[1])}"
            )
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise IntegrityError("Requested raw X rows contain non-finite or negative values")
        if np.any(np.abs(values - np.rint(values)) > 1.0e-6):
            raise IntegrityError(
                "Requested X rows are not raw integer-valued UMI pseudobulk counts"
            )
        return values


@dataclass(frozen=True)
class FrozenState:
    current_head: str
    branch: str
    remote_heads: dict[str, str]
    source_lock: dict
    run_status: dict
    model_lock: dict
    frozen_input_sha256: dict[str, str]
    builder_sha256: str


def verify_dual_remote_contains_head(current_head: str) -> tuple[str, dict[str, str]]:
    branch = git_text("rev-parse", "--abbrev-ref", "HEAD")
    if branch == "HEAD":
        raise IntegrityError("E168 expression access requires a named Git branch")
    remote_heads: dict[str, str] = {}
    for remote in ("origin", "github"):
        fetched_ref = f"refs/remotes/{remote}/{branch}"
        result = git(
            "fetch", "--quiet", remote,
            f"refs/heads/{branch}:{fetched_ref}", check=False,
        )
        if result.returncode:
            raise IntegrityError(
                f"Could not verify code freeze on remote {remote}: "
                f"{result.stderr.decode('utf-8', errors='replace').strip()}"
            )
        remote_head = git_text("rev-parse", fetched_ref)
        if git("merge-base", "--is-ancestor", current_head, remote_head, check=False).returncode:
            raise IntegrityError(
                f"Current code HEAD {current_head} is absent from {remote}/{branch}"
            )
        remote_heads[remote] = remote_head
    return branch, remote_heads


def verify_frozen_state() -> FrozenState:
    try:
        git("cat-file", "-e", f"{FROZEN_METADATA_COMMIT}^{{commit}}")
    except subprocess.CalledProcessError as exc:
        raise IntegrityError("Frozen E168 metadata commit is unavailable locally") from exc
    current_head = git_text("rev-parse", "HEAD")
    if git("merge-base", "--is-ancestor", FROZEN_METADATA_COMMIT, current_head, check=False).returncode:
        raise IntegrityError(
            f"Current HEAD {current_head} does not descend from frozen E168 commit "
            f"{FROZEN_METADATA_COMMIT}"
        )

    observed: dict[str, str] = {}
    for relative in FROZEN_INPUTS:
        local = ROOT / relative
        if not local.is_file():
            raise IntegrityError(f"Frozen input is missing: {relative}")
        try:
            frozen_bytes = git("show", f"{FROZEN_METADATA_COMMIT}:{relative.as_posix()}").stdout
        except subprocess.CalledProcessError as exc:
            raise IntegrityError(f"Frozen Git blob is missing: {relative}") from exc
        local_bytes = local.read_bytes()
        if frozen_bytes != local_bytes:
            raise IntegrityError(f"Frozen input differs from commit {FROZEN_METADATA_COMMIT}: {relative}")
        observed[relative.as_posix()] = sha256_bytes(local_bytes)

    run_status = json.loads((ROOT / RUN_STATUS_REL).read_text(encoding="utf-8"))
    if run_status.get("status") != "PASS" or run_status.get("stage") != "F1_METADATA_FREEZE":
        raise IntegrityError("E168 metadata freeze did not have the registered PASS state")
    for relative, expected in run_status.get("artifact_sha256", {}).items():
        path = EXPERIMENT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise IntegrityError(f"Metadata-freeze artifact hash changed: {relative}")

    builder = ROOT / BUILDER_REL
    try:
        committed_builder = git("show", f"HEAD:{BUILDER_REL.as_posix()}").stdout
    except subprocess.CalledProcessError as exc:
        raise IntegrityError("Asset builder must be committed before it can read X") from exc
    if builder.read_bytes() != committed_builder:
        raise IntegrityError("Working-tree asset builder differs from current HEAD")
    branch, remote_heads = verify_dual_remote_contains_head(current_head)

    source_lock = json.loads((ROOT / SOURCE_LOCK_REL).read_text(encoding="utf-8"))
    model_lock = json.loads((ROOT / MODEL_LOCK_REL).read_text(encoding="utf-8"))
    return FrozenState(
        current_head=current_head,
        branch=branch,
        remote_heads=remote_heads,
        source_lock=source_lock,
        run_status=run_status,
        model_lock=model_lock,
        frozen_input_sha256=observed,
        builder_sha256=sha256_file(builder),
    )


def validate_manifests() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = pd.read_csv(ROOT / ROW_ACCESS_REL, keep_default_na=False)
    targets = pd.read_csv(ROOT / TARGETS_REL, keep_default_na=False)
    tasks = pd.read_csv(ROOT / TASKS_REL, keep_default_na=False)
    roles = pd.read_csv(ROOT / DONOR_ROLES_REL, keep_default_na=False)

    required_row_columns = {
        "metadata_row_index", "donor_id", "donor_role", "culture_condition",
        "guide_id", "guide_type_normalized", "ensembl_core", "target_stratum",
        "x_access_phase",
    }
    if not required_row_columns.issubset(rows.columns):
        raise IntegrityError(f"Row-access manifest schema changed: {list(rows.columns)}")
    rows["metadata_row_index"] = pd.to_numeric(rows.metadata_row_index, errors="raise").astype(np.int64)
    if rows.metadata_row_index.duplicated().any():
        raise IntegrityError("Row-access manifest maps one source row more than once")
    counts = rows.x_access_phase.value_counts().to_dict()
    if counts != EXPECTED_PHASE_COUNTS:
        raise IntegrityError(f"Frozen row-access phase counts changed: {counts}")
    if len(targets) != N_TARGET_GENES or targets.ensembl_core.nunique() != N_TARGET_GENES:
        raise IntegrityError("Frozen selected-target manifest is not exactly 200 unique targets")
    if targets.target_stratum.value_counts().to_dict() != {
        "DONOR_UNSEEN_ONLY": 160, "COLUMN_UNSEEN": 40
    }:
        raise IntegrityError("Frozen 160/40 target stratification changed")
    if len(tasks) != 2400 or tasks.task_id.nunique() != 2400:
        raise IntegrityError("Frozen task manifest is not exactly 2,400 unique tasks")
    if len(roles) != 12 or set(roles.culture_condition) != set(EXPECTED_STATES):
        raise IntegrityError("Frozen donor/state role table changed")
    if roles.donor_role.value_counts().to_dict() != {"train": 6, "validation": 3, "test": 3}:
        raise IntegrityError("Frozen donor role counts changed")
    return rows, targets, tasks, roles


def verify_complete_source(frozen: FrozenState) -> tuple[Path, str]:
    source = Path(frozen.source_lock["local_target_path"])
    if source != DATA_ROOT / "source/GWCD4i.pseudobulk_merged.h5ad":
        raise IntegrityError(f"Unexpected frozen source path: {source}")
    if not source.is_file():
        raise IntegrityError(f"Complete source H5AD is missing: {source}")
    if source.is_symlink():
        raise IntegrityError("Frozen source H5AD must not be a symbolic link")
    expected_size = int(frozen.source_lock["content_length_bytes"])
    actual_size = source.stat().st_size
    if actual_size != expected_size:
        raise IntegrityError(f"Source byte length is incomplete: {actual_size} != {expected_size}")
    aria2_files = sorted(source.parent.glob("*.aria2"))
    explicit_sidecar = Path(str(source) + ".aria2")
    if explicit_sidecar.exists() and explicit_sidecar not in aria2_files:
        aria2_files.append(explicit_sidecar)
    if aria2_files:
        raise IntegrityError(f"aria2 download sidecar still exists: {aria2_files}")
    if not BYTE_ATTESTATION.is_file() or BYTE_ATTESTATION.is_symlink():
        raise IntegrityError(
            "Official-checksum byte attestation is missing; assemble with the frozen "
            "E168 downloader before opening HDF5"
        )
    byte_attestation = json.loads(BYTE_ATTESTATION.read_text(encoding="utf-8"))
    expected_crc64 = frozen.source_lock["checksum_crc64nvme_base64"]
    required_attestation = {
        "assembled_path": str(source),
        "assembled_size": expected_size,
        "computed_crc64nvme_base64": expected_crc64,
        "official_crc64nvme_base64": expected_crc64,
        "crc64nvme_matches_official_full_object_checksum": True,
        "hdf5_opened": False,
        "expression_values_decoded": False,
    }
    mismatches = {
        key: {"expected": value, "observed": byte_attestation.get(key)}
        for key, value in required_attestation.items()
        if byte_attestation.get(key) != value
    }
    if mismatches:
        raise IntegrityError(f"Source byte attestation mismatch: {mismatches}")
    code_freeze = byte_attestation.get("code_freeze", {})
    attested_download_head = str(code_freeze.get("git_head", ""))
    attested_downloader_sha = str(code_freeze.get("downloader_sha256", ""))
    if code_freeze.get("downloader_path") != DOWNLOADER_REL.as_posix():
        raise IntegrityError("Source byte attestation names an unexpected downloader")
    if git("cat-file", "-e", f"{attested_download_head}^{{commit}}", check=False).returncode:
        raise IntegrityError("Attested downloader commit is unavailable")
    if git(
        "merge-base", "--is-ancestor", attested_download_head, frozen.current_head,
        check=False,
    ).returncode:
        raise IntegrityError("Current code does not descend from the attested downloader commit")
    committed_downloader = git(
        "show", f"{attested_download_head}:{DOWNLOADER_REL.as_posix()}"
    ).stdout
    if sha256_bytes(committed_downloader) != attested_downloader_sha:
        raise IntegrityError("Attested downloader SHA-256 does not match its Git commit")
    if sha256_file(ROOT / DOWNLOADER_REL) != attested_downloader_sha:
        raise IntegrityError("Downloader changed after official-checksum assembly")
    attested_sha256 = str(byte_attestation.get("sha256", ""))
    if len(attested_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in attested_sha256
    ):
        raise IntegrityError("Source byte attestation has no valid full-file SHA-256")
    # This is intentionally a whole-file pass before any HDF5 X access.  It
    # binds the current bytes to the assembly that matched S3's full-object
    # CRC64NVME checksum rather than trusting length or multipart ETag alone.
    observed_sha256 = sha256_file(source)
    if observed_sha256 != attested_sha256:
        raise IntegrityError(
            "Source bytes changed after official CRC64NVME-verified assembly: "
            f"{observed_sha256} != {attested_sha256}"
        )
    return source, observed_sha256


def clean_ensembl(value: object) -> str:
    return str(value).strip().split(".", 1)[0]


def load_gene_axis(handle: h5py.File, n_vars: int) -> pd.DataFrame:
    gene_ids = read_categorical_or_array(handle, "var/gene_ids")
    gene_names = read_categorical_or_array(handle, "var/gene_name")
    if len(gene_ids) != n_vars or len(gene_names) != n_vars:
        raise IntegrityError("Frozen var identity axes do not match X shape")
    axis = pd.DataFrame(
        {
            "source_column_index": np.arange(n_vars, dtype=np.int64),
            "ensembl_id": [clean_ensembl(value) for value in gene_ids],
            "gene_name": [str(value).strip() for value in gene_names],
        }
    )
    if axis.ensembl_id.eq("").any() or axis.ensembl_id.duplicated().any():
        raise IntegrityError("Expression-axis Ensembl identifiers are empty or duplicated")
    return axis


def iter_batches(frame: pd.DataFrame, batch_size: int) -> Iterable[pd.DataFrame]:
    ordered = frame.sort_values("metadata_row_index", kind="stable").reset_index(drop=True)
    for start in range(0, len(ordered), batch_size):
        yield ordered.iloc[start : start + batch_size].copy()


def consume_control_rows(
    reader: RowMatrixReader,
    rows: pd.DataFrame,
    batch_size: int,
) -> tuple[
    dict[str, np.ndarray],
    dict[tuple[str, str, str], np.ndarray],
    dict[tuple[str, str, str], float],
    list[dict],
]:
    control_rows = rows.loc[rows.x_access_phase.eq("PRETRUTH_CONTROL_X")].copy()
    raw_by_context: dict[str, np.ndarray] = {}
    train_guide_raw: dict[tuple[str, str, str], np.ndarray] = {}
    train_guide_library_sum: defaultdict[tuple[str, str, str], float] = defaultdict(float)
    access: list[dict] = []
    for batch in iter_batches(control_rows, batch_size):
        indices = batch.metadata_row_index.astype(int).tolist()
        values = reader.read_rows(indices)
        for metadata, vector in zip(batch.itertuples(index=False), values, strict=True):
            key = f"{metadata.donor_id}::{metadata.culture_condition}"
            if key not in raw_by_context:
                raw_by_context[key] = np.zeros(reader.shape[1], dtype=np.float64)
            raw_by_context[key] += vector
            if str(metadata.donor_role) == "train":
                guide_key = (
                    str(metadata.donor_id),
                    str(metadata.culture_condition),
                    str(metadata.guide_id),
                )
                if guide_key not in train_guide_raw:
                    train_guide_raw[guide_key] = np.zeros(reader.shape[1], dtype=np.float64)
                train_guide_raw[guide_key] += vector
                train_guide_library_sum[guide_key] += float(vector.sum(dtype=np.float64))
            access.append(
                {
                    "metadata_row_index": int(metadata.metadata_row_index),
                    "x_access_phase": str(metadata.x_access_phase),
                    "logical_x_row_read_count": 1,
                    "asset_stage": "F2_PRETRUTH",
                    "purpose": "matched_non_targeting_control",
                }
            )
    if len(raw_by_context) != 12:
        raise IntegrityError(f"Expected 12 donor/state NTC contexts, observed {len(raw_by_context)}")
    if not train_guide_raw:
        raise IntegrityError("No train-donor NTC guide profiles were found")
    return raw_by_context, train_guide_raw, dict(train_guide_library_sum), access


def build_train_ntc_coexpression(
    train_guide_raw: Mapping[tuple[str, str, str], np.ndarray],
    train_guide_library_sum: Mapping[tuple[str, str, str], float],
    panel: pd.DataFrame,
    k: int = 10,
    threshold: float = 0.4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce E112's absolute-correlation, top-k control graph rule."""
    panel_columns = panel.source_column_index.to_numpy(dtype=np.int64)
    keys = sorted(train_guide_raw)
    profiles = np.vstack(
        [
            normalize_panel_from_merged_counts(
                train_guide_raw[key][panel_columns], train_guide_library_sum[key]
            )
            for key in keys
        ]
    )
    if profiles.shape[0] < 2 or profiles.shape[1] != PANEL_SIZE:
        raise IntegrityError(f"Invalid train NTC guide matrix for coexpression: {profiles.shape}")
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(np.asarray(profiles, dtype=np.float32), rowvar=False)
    corr = np.nan_to_num(np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0)
    genes = panel.scgpt_token.astype(str).tolist()
    rows: list[dict] = []
    for target_index, target in enumerate(genes):
        # This intentionally matches E112, including its NumPy argsort rule and
        # inclusion of self even when the threshold is not met.
        candidates = np.argsort(corr[:, target_index])[-(k + 1) :][::-1].tolist()
        # A constant train-NTC gene has a zeroed diagonal after nan_to_num.
        # NumPy's top-k tie order can then omit its own index.  The registered
        # graph rule says every target keeps a self edge, so add that index
        # explicitly when it is absent instead of silently dropping the gene.
        if target_index not in candidates:
            candidates.append(target_index)
        for source_index in candidates:
            importance = float(corr[int(source_index), target_index])
            if importance >= threshold or int(source_index) == target_index:
                rows.append(
                    {
                        "source": genes[int(source_index)],
                        "target": target,
                        "importance": importance,
                        "source_panel_index": int(source_index),
                        "target_panel_index": int(target_index),
                        "absolute_correlation": True,
                        "top_k": int(k),
                        "threshold": float(threshold),
                    }
                )
    edges = pd.DataFrame(rows)
    if edges.empty or set(genes) - set(edges.target):
        raise IntegrityError("Train-only NTC coexpression graph is missing target genes")
    profile_index = pd.DataFrame(
        [
            {
                "profile_index": index,
                "donor_id": key[0],
                "culture_condition": key[1],
                "ntc_guide_id": key[2],
                "raw_library_sum": float(train_guide_library_sum[key]),
                "used_for_train_only_coexpression": True,
            }
            for index, key in enumerate(keys)
        ]
    )
    return edges, profile_index


def build_panel(
    axis: pd.DataFrame,
    targets: pd.DataFrame,
    raw_controls: Mapping[str, np.ndarray],
    roles: pd.DataFrame,
    vocab_path: Path,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    normalized_controls = {
        key: normalize_merged_raw_counts(value) for key, value in raw_controls.items()
    }
    train_keys = [
        f"{row.donor_id}::{row.culture_condition}"
        for row in roles.loc[roles.donor_role.eq("train")].itertuples(index=False)
    ]
    if len(train_keys) != 6 or any(key not in normalized_controls for key in train_keys):
        raise IntegrityError("The six frozen training NTC contexts are not all available")
    train_ntc_mean = np.mean([normalized_controls[key] for key in train_keys], axis=0)

    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    vocab_tokens = set(map(str, vocab.keys())) if isinstance(vocab, dict) else set(map(str, vocab))
    axis = axis.copy()
    axis["train_ntc_mean_expression"] = train_ntc_mean
    by_ensembl = axis.set_index("ensembl_id", drop=False)

    panel_rows: list[dict] = []
    used_ensembl: set[str] = set()
    used_tokens: set[str] = set()
    for target in targets.sort_values("target_selection_rank", kind="stable").itertuples(index=False):
        ensembl = clean_ensembl(target.ensembl_core)
        if ensembl not in by_ensembl.index:
            raise IntegrityError(f"Selected target is missing from expression axis: {ensembl}")
        source = by_ensembl.loc[ensembl]
        if isinstance(source, pd.DataFrame):
            raise IntegrityError(f"Selected target maps to multiple source columns: {ensembl}")
        if str(source.gene_name) != str(target.scgpt_token):
            raise IntegrityError(
                f"Frozen target token no longer matches expression axis: {ensembl}"
            )
        if str(target.scgpt_token) not in vocab_tokens:
            raise IntegrityError(f"Selected target token disappeared from scGPT vocab: {ensembl}")
        used_ensembl.add(ensembl)
        used_tokens.add(str(target.scgpt_token))
        panel_rows.append(
            {
                "source_column_index": int(source.source_column_index),
                "ensembl_id": ensembl,
                "gene_name": str(source.gene_name),
                "scgpt_token": str(target.scgpt_token),
                "panel_role": "REGISTERED_TARGET",
                "target_selection_rank": int(target.target_selection_rank),
                "train_ntc_mean_expression": float(source.train_ntc_mean_expression),
            }
        )

    extras = axis.loc[
        ~axis.ensembl_id.isin(used_ensembl)
        & axis.gene_name.isin(vocab_tokens)
        & axis.gene_name.ne("")
    ].copy()
    extras = extras.sort_values(
        ["train_ntc_mean_expression", "ensembl_id"], ascending=[False, True], kind="stable"
    )
    for row in extras.itertuples(index=False):
        token = str(row.gene_name)
        if token in used_tokens:
            continue
        used_tokens.add(token)
        used_ensembl.add(str(row.ensembl_id))
        panel_rows.append(
            {
                "source_column_index": int(row.source_column_index),
                "ensembl_id": str(row.ensembl_id),
                "gene_name": token,
                "scgpt_token": token,
                "panel_role": "TRAIN_NTC_HIGH_EXPRESSION",
                "target_selection_rank": "",
                "train_ntc_mean_expression": float(row.train_ntc_mean_expression),
            }
        )
        if len(panel_rows) == PANEL_SIZE:
            break
    if len(panel_rows) != PANEL_SIZE:
        raise IntegrityError(f"Could select only {len(panel_rows)} of {PANEL_SIZE} panel genes")
    panel = pd.DataFrame(panel_rows)
    panel.insert(0, "panel_index", np.arange(PANEL_SIZE, dtype=np.int64))
    if panel.ensembl_id.nunique() != PANEL_SIZE or panel.scgpt_token.nunique() != PANEL_SIZE:
        raise IntegrityError("Frozen panel is not one-to-one in Ensembl ID and scGPT token")
    panel_columns = panel.source_column_index.to_numpy(dtype=np.int64)
    panel_controls = {
        key: np.asarray(profile[panel_columns], dtype=np.float64)
        for key, profile in normalized_controls.items()
    }
    return panel, panel_controls


def expected_guides_by_target(targets: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for row in targets.itertuples(index=False):
        guides = tuple(sorted(filter(None, str(row.eligible_guide_ids).split("+"))))
        if len(guides) != int(row.n_identity_complete_guides) or len(guides) < 1:
            raise IntegrityError(f"Frozen eligible guide list is invalid: {row.ensembl_core}")
        result[clean_ensembl(row.ensembl_core)] = guides
    return result


def consume_target_rows(
    reader: RowMatrixReader,
    selected_rows: pd.DataFrame,
    panel_columns: np.ndarray,
    controls: Mapping[str, np.ndarray],
    expected_guides: Mapping[str, tuple[str, ...]],
    valid_task_ids: set[str],
    stage: str,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict], pd.DataFrame]:
    # A guide may in principle span multiple source runs.  Sum raw counts first;
    # only then normalize, exactly as preregistered.
    guide_raw_panel: dict[tuple[str, str], np.ndarray] = {}
    guide_library_sum: defaultdict[tuple[str, str], float] = defaultdict(float)
    guide_row_count: defaultdict[tuple[str, str], int] = defaultdict(int)
    guide_metadata: dict[tuple[str, str], dict] = {}
    access: list[dict] = []

    for batch in iter_batches(selected_rows, batch_size):
        indices = batch.metadata_row_index.astype(int).tolist()
        values = reader.read_rows(indices)
        for metadata, vector in zip(batch.itertuples(index=False), values, strict=True):
            ensembl = clean_ensembl(metadata.ensembl_core)
            task_id = f"{TASK_PREFIX}::{metadata.donor_id}::{metadata.culture_condition}::{ensembl}"
            if task_id not in valid_task_ids:
                raise IntegrityError(f"X row does not map to a frozen task: {task_id}")
            key = (task_id, str(metadata.guide_id))
            panel_values = np.asarray(vector[panel_columns], dtype=np.float64)
            if key not in guide_raw_panel:
                guide_raw_panel[key] = np.zeros(len(panel_columns), dtype=np.float64)
                guide_metadata[key] = {
                    "task_id": task_id,
                    "guide_id": str(metadata.guide_id),
                    "donor_id": str(metadata.donor_id),
                    "culture_condition": str(metadata.culture_condition),
                    "ensembl_id": ensembl,
                    "target_stratum": str(metadata.target_stratum),
                    "x_access_phase": str(metadata.x_access_phase),
                }
            elif guide_metadata[key]["x_access_phase"] != str(metadata.x_access_phase):
                raise IntegrityError(f"One guide crosses access phases: {key}")
            guide_raw_panel[key] += panel_values
            guide_library_sum[key] += float(vector.sum(dtype=np.float64))
            guide_row_count[key] += 1
            access.append(
                {
                    "metadata_row_index": int(metadata.metadata_row_index),
                    "x_access_phase": str(metadata.x_access_phase),
                    "logical_x_row_read_count": 1,
                    "asset_stage": stage,
                    "purpose": "registered_targeting_guide",
                }
            )

    guide_effects: dict[str, np.ndarray] = {}
    guide_table_rows: list[dict] = []
    task_to_guides: defaultdict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
    for key in sorted(guide_raw_panel):
        task_id, guide_id = key
        metadata = guide_metadata[key]
        context_key = f"{metadata['donor_id']}::{metadata['culture_condition']}"
        if context_key not in controls:
            raise IntegrityError(f"Matched frozen NTC context is absent: {context_key}")
        profile = normalize_panel_from_merged_counts(
            guide_raw_panel[key], guide_library_sum[key]
        )
        effect = profile - np.asarray(controls[context_key], dtype=np.float64)
        asset_key = f"{task_id}::{guide_id}"
        guide_effects[asset_key] = effect
        task_to_guides[task_id].append((guide_id, effect))
        guide_table_rows.append(
            {
                **metadata,
                "guide_effect_asset_key": asset_key,
                "source_rows_merged_before_normalization": int(guide_row_count[key]),
                "raw_library_sum": float(guide_library_sum[key]),
            }
        )

    target_effects: dict[str, np.ndarray] = {}
    for task_id in sorted(task_to_guides):
        ensembl = task_id.rsplit("::", 1)[-1]
        observed = tuple(sorted(guide for guide, _ in task_to_guides[task_id]))
        expected = expected_guides.get(ensembl)
        if observed != expected:
            raise IntegrityError(
                f"Task guide set changed for {task_id}: observed={observed}, expected={expected}"
            )
        # Equal guide weights are fixed; no abundance or consistency weighting.
        target_effects[task_id] = np.mean(
            [effect for _, effect in sorted(task_to_guides[task_id])], axis=0
        )
    return target_effects, guide_effects, access, pd.DataFrame(guide_table_rows)


def save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    if not arrays:
        raise IntegrityError(f"Refusing to write an empty array asset: {path.name}")
    ordered = {key: np.asarray(arrays[key], dtype=np.float64) for key in sorted(arrays)}
    np.savez_compressed(path, **ordered)
    with np.load(path, allow_pickle=False) as check:
        if set(check.files) != set(ordered):
            raise IntegrityError(f"NPZ round-trip key mismatch: {path}")
        for key, expected in ordered.items():
            actual = check[key]
            if actual.shape != expected.shape or not np.array_equal(actual, expected):
                raise IntegrityError(f"NPZ round-trip value mismatch: {path}:{key}")


def write_manifest(directory: Path) -> tuple[dict[str, str], str]:
    hashes: dict[str, str] = {}
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "MANIFEST.sha256":
            hashes[path.name] = sha256_file(path)
    text = "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items()))
    atomic_write_text(directory / "MANIFEST.sha256", text)
    return hashes, sha256_file(directory / "MANIFEST.sha256")


def prepare_staging(destination: Path) -> Path:
    ISOLATED_ROOT.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(destination.name + f".staging.{os.getpid()}")
    if destination.exists():
        raise IntegrityError(f"Refusing to overwrite immutable isolated asset: {destination}")
    stale = list(ISOLATED_ROOT.glob(destination.name + ".staging.*"))
    if stale:
        raise IntegrityError(f"Staging directory already exists; audit/remove it first: {stale}")
    staging.mkdir(mode=0o750)
    return staging


def complete_staging(staging: Path, destination: Path) -> None:
    os.replace(staging, destination)


def assert_exact_access(
    access: pd.DataFrame,
    expected_rows: pd.DataFrame,
    allowed_phases: Sequence[str],
) -> dict[str, int]:
    if access.empty:
        raise IntegrityError("No X rows were recorded in the access audit")
    if access.metadata_row_index.duplicated().any():
        raise IntegrityError("At least one source X row was read more than once")
    observed_indices = set(access.metadata_row_index.astype(int))
    expected = expected_rows.loc[
        expected_rows.x_access_phase.isin(allowed_phases), "metadata_row_index"
    ].astype(int)
    if observed_indices != set(expected):
        missing = sorted(set(expected) - observed_indices)[:10]
        extra = sorted(observed_indices - set(expected))[:10]
        raise IntegrityError(f"X access did not equal frozen rows; missing={missing}, extra={extra}")
    phases = access.x_access_phase.value_counts().to_dict()
    expected_counts = {
        phase: EXPECTED_PHASE_COUNTS[phase] for phase in allowed_phases
    }
    if phases != expected_counts:
        raise IntegrityError(f"X access phase counts changed: {phases} != {expected_counts}")
    return {key: int(value) for key, value in phases.items()}


def build_pretruth(batch_size: int) -> Path:
    frozen = verify_frozen_state()
    rows, targets, tasks, roles = validate_manifests()
    source, source_sha256 = verify_complete_source(frozen)
    staging = prepare_staging(F2_DIR)
    try:
        with h5py.File(source, "r") as handle:
            reader = RowMatrixReader(handle)
            expected_shape = (
                int(frozen.run_status["source_schema"]["n_obs"]),
                int(frozen.run_status["source_schema"]["n_vars"]),
            )
            if reader.shape != expected_shape:
                raise IntegrityError(f"Source X shape changed: {reader.shape} != {expected_shape}")
            axis = load_gene_axis(handle, reader.shape[1])
            (
                raw_controls,
                train_guide_raw,
                train_guide_library_sum,
                control_access,
            ) = consume_control_rows(reader, rows, batch_size)
            vocab_path = Path(
                next(
                    path for path in frozen.model_lock["scgpt_checkpoint_files"]
                    if path.endswith("/vocab.json")
                )
            )
            if sha256_file(vocab_path) != frozen.model_lock["scgpt_checkpoint_files"][str(vocab_path)]:
                raise IntegrityError("Frozen scGPT vocabulary hash changed")
            panel, controls = build_panel(axis, targets, raw_controls, roles, vocab_path)
            coexpression_edges, coexpression_profiles = build_train_ntc_coexpression(
                train_guide_raw, train_guide_library_sum, panel, k=10, threshold=0.4
            )
            # Release the only large in-memory control-row cache before reading
            # targeting rows. It is never exported to the model runner.
            del train_guide_raw
            panel_columns = panel.source_column_index.to_numpy(dtype=np.int64)
            target_rows = rows.loc[rows.x_access_phase.isin(
                ("PRETRUTH_TRAIN_X", "PRETRUTH_VALIDATION_X")
            )].copy()
            seen_effects, guide_effects, target_access, guide_table = consume_target_rows(
                reader=reader,
                selected_rows=target_rows,
                panel_columns=panel_columns,
                controls=controls,
                expected_guides=expected_guides_by_target(targets),
                valid_task_ids=set(tasks.task_id.astype(str)),
                stage="F2_PRETRUTH",
                batch_size=batch_size,
            )

        if len(seen_effects) != 1440 or len(guide_effects) != 2880:
            raise IntegrityError(
                f"Unexpected pretruth effect counts: tasks={len(seen_effects)}, guides={len(guide_effects)}"
            )
        forbidden_pretruth_tasks = set(
            tasks.loc[
                tasks.donor_role.eq("test")
                | tasks.target_stratum.eq("COLUMN_UNSEEN"),
                "task_id",
            ].astype(str)
        )
        leaked = sorted(set(seen_effects) & forbidden_pretruth_tasks)
        if leaked:
            raise IntegrityError(
                "Test-donor or column-unseen targeting truth leaked into pretruth "
                f"effects: {leaked[:3]}"
            )
        if guide_table.target_stratum.ne("DONOR_UNSEEN_ONLY").any():
            raise IntegrityError("Column-unseen train/validation truth leaked into pretruth")

        panel.to_csv(staging / "GENE_PANEL.csv", index=False, float_format="%.17g")
        coexpression_edges.to_csv(
            staging / "TRAIN_NTC_COEXPRESSION_EDGES.csv", index=False, float_format="%.17g"
        )
        coexpression_profiles.to_csv(
            staging / "TRAIN_NTC_COEXPRESSION_PROFILE_INDEX.csv", index=False,
            float_format="%.17g",
        )
        save_npz(staging / "CONTROL_PROFILES.npz", controls)
        save_npz(staging / "SEEN_TARGET_EFFECTS.npz", seen_effects)
        pretruth_tasks = tasks.copy()
        pretruth_tasks["effect_asset_key"] = np.where(
            pretruth_tasks.task_id.isin(seen_effects), pretruth_tasks.task_id, ""
        )
        pretruth_tasks["pretruth_x_access"] = np.where(
            pretruth_tasks.task_id.isin(seen_effects), "SEEN_TARGET_EFFECT_AVAILABLE", "QUERY_ONLY"
        )
        pretruth_tasks.to_csv(staging / "PRETRUTH_TASKS.csv", index=False)
        guide_table.to_csv(staging / "PRETRUTH_GUIDE_EFFECT_INDEX.csv", index=False)
        access = pd.DataFrame(control_access + target_access).sort_values(
            "metadata_row_index", kind="stable"
        )
        phase_counts = assert_exact_access(access, rows, PRETRUTH_PHASES)
        access.to_csv(staging / "ROW_ACCESS_AUDIT.csv", index=False)

        primary_hashes = {
            path.name: sha256_file(path)
            for path in staging.iterdir() if path.is_file()
        }
        attestation = {
            "experiment": EXPERIMENT_ID,
            "stage": PRETRUTH_ASSET_STAGE,
            "status": "PASS",
            "deployment_authorized": False,
            "frozen_metadata_commit": FROZEN_METADATA_COMMIT,
            "current_git_head": frozen.current_head,
            "code_freeze_branch": frozen.branch,
            "code_freeze_remote_heads": frozen.remote_heads,
            "builder_sha256": frozen.builder_sha256,
            "frozen_input_sha256": frozen.frozen_input_sha256,
            "source_path": str(source),
            "source_bytes": source.stat().st_size,
            "source_full_sha256": source_sha256,
            "source_official_crc64nvme_base64": frozen.source_lock["checksum_crc64nvme_base64"],
            "source_byte_attestation_sha256": sha256_file(BYTE_ATTESTATION),
            "source_full_sha256_computed_before_x_access": True,
            "source_download_sidecar_present": False,
            "x_encoding": reader.encoding,
            "x_shape": list(reader.shape),
            "logical_x_rows_read": int(len(access)),
            "logical_x_rows_read_by_phase": phase_counts,
            "all_returned_x_rows_read_exactly_once": True,
            "test_targeting_x_values_read": 0,
            "pretruth_test_targeting_x_values_read": 0,
            "forbidden_column_unseen_x_values_read": 0,
            "postgate_test_targeting_x_values_read": 0,
            "n_panel_genes": len(panel),
            "n_registered_target_genes_in_panel": int(panel.panel_role.eq("REGISTERED_TARGET").sum()),
            "n_train_ntc_selected_extra_genes": int(panel.panel_role.eq("TRAIN_NTC_HIGH_EXPRESSION").sum()),
            "n_control_contexts": len(controls),
            "n_train_ntc_guide_profiles_used_for_coexpression": len(coexpression_profiles),
            "n_train_only_coexpression_edges": len(coexpression_edges),
            "coexpression_rule": "E112_absolute_Pearson_top10_threshold_0.4_self_included",
            "coexpression_uses_validation_or_test_ntc": False,
            "raw_ntc_guide_profiles_exported": False,
            "n_seen_train_validation_target_effects": len(seen_effects),
            "n_seen_train_validation_guide_effects_computed_in_memory": len(guide_effects),
            "guide_weighting": "equal_after_per_guide_raw_sum_then_log1p_1e4_normalization",
            "control_definition": "merge_all_NTC_raw_counts_within_donor_state_then_log1p_1e4",
            "test_performance_metrics_computed": 0,
            "primary_output_sha256": primary_hashes,
        }
        atomic_write_text(
            staging / "ACCESS_ATTESTATION.json",
            json.dumps(attestation, indent=2, sort_keys=True) + "\n",
        )
        write_manifest(staging)
        # Verify that the F2 directory contains no raw source or postgate truth by name.
        forbidden_names = [
            path.name for path in staging.iterdir()
            if "POSTGATE" in path.name.upper() or path.suffix.lower() in {".h5", ".h5ad"}
        ]
        if forbidden_names:
            raise IntegrityError(f"Forbidden file in F2 isolated directory: {forbidden_names}")
        complete_staging(staging, F2_DIR)
        return F2_DIR
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_manifest(directory: Path) -> str:
    manifest = directory / "MANIFEST.sha256"
    if not manifest.is_file():
        raise IntegrityError(f"Isolated manifest is absent: {manifest}")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        path = directory / name
        if not path.is_file() or sha256_file(path) != digest:
            raise IntegrityError(f"Isolated asset hash mismatch: {path}")
    return sha256_file(manifest)


def verify_gate_snapshot(
    snapshot_path: Path,
    gate_commit: str,
    branch: str,
) -> tuple[dict, dict[str, str]]:
    try:
        relative = snapshot_path.resolve().relative_to(ROOT.resolve())
    except ValueError as exc:
        raise IntegrityError("Gate snapshot must be a committed file inside the repository") from exc
    local_bytes = snapshot_path.read_bytes()
    try:
        committed = git("show", f"{gate_commit}:{relative.as_posix()}").stdout
    except subprocess.CalledProcessError as exc:
        raise IntegrityError("Gate snapshot is not present in the supplied gate commit") from exc
    if committed != local_bytes:
        raise IntegrityError("Local gate snapshot differs from its committed bytes")
    if git("merge-base", "--is-ancestor", gate_commit, "HEAD", check=False).returncode:
        raise IntegrityError("Current HEAD does not contain the supplied gate commit")

    remote_heads: dict[str, str] = {}
    for remote in ("origin", "github"):
        fetched_ref = f"refs/remotes/{remote}/{branch}"
        result = git(
            "fetch", "--quiet", remote,
            f"refs/heads/{branch}:{fetched_ref}", check=False,
        )
        if result.returncode:
            raise IntegrityError(
                f"Could not verify committed pretruth gate on remote {remote}: "
                f"{result.stderr.decode('utf-8', errors='replace').strip()}"
            )
        remote_head = git_text("rev-parse", fetched_ref)
        if git("merge-base", "--is-ancestor", gate_commit, remote_head, check=False).returncode:
            raise IntegrityError(
                f"Remote {remote}/{branch} does not contain gate commit {gate_commit}"
            )
        remote_heads[remote] = remote_head

    snapshot = json.loads(local_bytes)
    required = {
        "experiment": EXPERIMENT_ID,
        "stage": PRETRUTH_GATE_STAGE,
        "status": "PASS",
        "all_registered_gates_passed": True,
        "test_targeting_x_values_read": 0,
        "forbidden_column_unseen_x_values_read": 0,
    }
    mismatches = {
        key: {"expected": value, "observed": snapshot.get(key)}
        for key, value in required.items() if snapshot.get(key) != value
    }
    if mismatches:
        raise IntegrityError(f"Pretruth gate snapshot is not an exact PASS: {mismatches}")
    return snapshot, remote_heads


def build_postgate(
    batch_size: int,
    snapshot_path: Path,
    gate_commit: str,
    branch: str,
) -> Path:
    frozen = verify_frozen_state()
    rows, targets, tasks, _roles = validate_manifests()
    source, source_sha256 = verify_complete_source(frozen)
    f2_manifest_sha = verify_manifest(F2_DIR)
    f2_attestation = json.loads((F2_DIR / "ACCESS_ATTESTATION.json").read_text(encoding="utf-8"))
    if f2_attestation.get("status") != "PASS":
        raise IntegrityError("F2 asset attestation is not PASS")
    if f2_attestation.get("source_full_sha256") != source_sha256:
        raise IntegrityError("Source bytes differ from the F2 asset build")
    snapshot, remote_heads = verify_gate_snapshot(snapshot_path, gate_commit, branch)
    if snapshot.get("source_full_sha256") != source_sha256:
        raise IntegrityError("Gate snapshot does not bind the full source SHA-256")
    if snapshot.get("f2_manifest_sha256") != f2_manifest_sha:
        raise IntegrityError("Gate snapshot does not bind the immutable F2 manifest")

    staging = prepare_staging(F3_DIR)
    try:
        panel = pd.read_csv(F2_DIR / "GENE_PANEL.csv", keep_default_na=False)
        if len(panel) != PANEL_SIZE or panel.panel_index.tolist() != list(range(PANEL_SIZE)):
            raise IntegrityError("F2 panel schema/order changed")
        panel_columns = panel.source_column_index.to_numpy(dtype=np.int64)
        with np.load(F2_DIR / "CONTROL_PROFILES.npz", allow_pickle=False) as control_asset:
            controls = {key: np.asarray(control_asset[key], dtype=np.float64) for key in control_asset.files}
        test_rows = rows.loc[rows.x_access_phase.eq("POSTGATE_TEST_TRUTH_X")].copy()
        with h5py.File(source, "r") as handle:
            reader = RowMatrixReader(handle)
            if max(panel_columns) >= reader.shape[1]:
                raise IntegrityError("F2 panel source-column mapping is outside X")
            test_effects, guide_effects, access_rows, guide_table = consume_target_rows(
                reader=reader,
                selected_rows=test_rows,
                panel_columns=panel_columns,
                controls=controls,
                expected_guides=expected_guides_by_target(targets),
                valid_task_ids=set(tasks.task_id.astype(str)),
                stage="F3_POSTGATE",
                batch_size=batch_size,
            )
        if len(test_effects) != 600 or len(guide_effects) != 1200:
            raise IntegrityError(
                f"Unexpected test truth counts: tasks={len(test_effects)}, guides={len(guide_effects)}"
            )
        if set(guide_table.x_access_phase) != {"POSTGATE_TEST_TRUTH_X"}:
            raise IntegrityError("Non-test or forbidden X entered the postgate truth asset")

        save_npz(staging / "TEST_TARGET_EFFECTS.npz", test_effects)
        save_npz(staging / "TEST_GUIDE_EFFECTS.npz", guide_effects)
        guide_table.to_csv(staging / "TEST_GUIDE_EFFECT_INDEX.csv", index=False)
        test_tasks = tasks.loc[tasks.primary_test_task.astype(str).str.lower().eq("true")].copy()
        if len(test_tasks) != 600:
            # Pandas normally inferred bool; handle that path without weakening the count.
            test_tasks = tasks.loc[tasks.primary_test_task == True].copy()  # noqa: E712
        if len(test_tasks) != 600 or set(test_tasks.task_id) != set(test_effects):
            raise IntegrityError("Frozen 600 test tasks do not match postgate truth keys")
        test_tasks["truth_effect_asset_key"] = test_tasks.task_id
        test_tasks.to_csv(staging / "TEST_TASKS.csv", index=False)
        access = pd.DataFrame(access_rows).sort_values("metadata_row_index", kind="stable")
        phase_counts = assert_exact_access(access, rows, POSTGATE_PHASES)
        access.to_csv(staging / "ROW_ACCESS_AUDIT.csv", index=False)

        primary_hashes = {
            path.name: sha256_file(path) for path in staging.iterdir() if path.is_file()
        }
        attestation = {
            "experiment": EXPERIMENT_ID,
            "stage": POSTGATE_ASSET_STAGE,
            "status": "PASS",
            "deployment_authorized": False,
            "frozen_metadata_commit": FROZEN_METADATA_COMMIT,
            "current_git_head": frozen.current_head,
            "code_freeze_branch": frozen.branch,
            "code_freeze_remote_heads": frozen.remote_heads,
            "builder_sha256": frozen.builder_sha256,
            "source_path": str(source),
            "source_bytes": source.stat().st_size,
            "source_full_sha256": source_sha256,
            "source_official_crc64nvme_base64": frozen.source_lock["checksum_crc64nvme_base64"],
            "source_byte_attestation_sha256": sha256_file(BYTE_ATTESTATION),
            "source_full_sha256_recomputed_before_postgate_x_access": True,
            "gate_snapshot_path": str(snapshot_path.resolve().relative_to(ROOT.resolve())),
            "gate_snapshot_sha256": sha256_file(snapshot_path),
            "gate_commit": gate_commit,
            "gate_remote_heads": remote_heads,
            "all_registered_pretruth_gates_passed": True,
            "f2_manifest_sha256": f2_manifest_sha,
            "logical_x_rows_read": len(access),
            "logical_x_rows_read_by_phase": phase_counts,
            "all_returned_x_rows_read_exactly_once": True,
            "postgate_test_targeting_x_values_read": len(access),
            "forbidden_column_unseen_x_values_read": 0,
            "train_or_validation_targeting_x_values_read_in_postgate": 0,
            "n_test_target_effects": len(test_effects),
            "n_test_guide_effects": len(guide_effects),
            "guide_weighting": "equal_after_per_guide_raw_sum_then_log1p_1e4_normalization",
            "test_performance_metrics_computed": 0,
            "primary_output_sha256": primary_hashes,
        }
        atomic_write_text(
            staging / "ACCESS_ATTESTATION.json",
            json.dumps(attestation, indent=2, sort_keys=True) + "\n",
        )
        write_manifest(staging)
        complete_staging(staging, F3_DIR)
        return F3_DIR
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def self_test() -> None:
    rng = np.random.default_rng(168)
    dense = rng.integers(0, 20, size=(11, 7), dtype=np.int64)
    with tempfile.TemporaryDirectory(prefix="e168_asset_builder_test_") as tmp:
        tmpdir = Path(tmp)
        dense_path = tmpdir / "dense.h5"
        with h5py.File(dense_path, "w") as handle:
            handle.create_dataset("X", data=dense)
        with h5py.File(dense_path, "r") as handle:
            reader = RowMatrixReader(handle)
            observed = reader.read_rows([0, 3, 10])
            if not np.array_equal(observed, dense[[0, 3, 10]]):
                raise AssertionError("Dense row reader failed")

        csr_path = tmpdir / "csr.h5"
        try:
            from scipy import sparse
        except ImportError as exc:  # pragma: no cover - production environment has scipy
            raise AssertionError("scipy is required for the CSR self-test") from exc
        matrix = sparse.csr_matrix(dense)
        with h5py.File(csr_path, "w") as handle:
            group = handle.create_group("X")
            group.attrs["encoding-type"] = "csr_matrix"
            group.attrs["shape"] = matrix.shape
            group.create_dataset("data", data=matrix.data)
            group.create_dataset("indices", data=matrix.indices)
            group.create_dataset("indptr", data=matrix.indptr)
        with h5py.File(csr_path, "r") as handle:
            reader = RowMatrixReader(handle)
            observed = reader.read_rows([1, 4, 9])
            if not np.array_equal(observed, dense[[1, 4, 9]]):
                raise AssertionError("CSR row reader failed")

        raw = np.asarray([0.0, 5.0, 15.0])
        expected = np.log1p(1.0e4 * raw / 20.0)
        if not np.allclose(normalize_merged_raw_counts(raw), expected):
            raise AssertionError("Normalization self-test failed")

        panel = pd.DataFrame(
            {
                "source_column_index": np.arange(PANEL_SIZE),
                "scgpt_token": [f"G{index}" for index in range(PANEL_SIZE)],
            }
        )
        identical_profile = np.arange(1, PANEL_SIZE + 1, dtype=np.float64)
        profiles = {
            ("D1", "S1", f"NTC{index}"): identical_profile.copy()
            for index in range(4)
        }
        library_sums = {key: float(value.sum()) for key, value in profiles.items()}
        edges, _profile_index = build_train_ntc_coexpression(
            profiles, library_sums, panel, k=10, threshold=0.4
        )
        if set(edges.target.astype(str)) != set(panel.scgpt_token.astype(str)):
            raise AssertionError("Constant-gene forced-self coexpression test failed")
        if not np.allclose(normalize_panel_from_merged_counts(raw[[0, 2]], 20.0), expected[[0, 2]]):
            raise AssertionError("Panel normalization self-test failed")
        # The production function fixes 512 columns.  Test its E112 edge rule
        # independently on a 512-column deterministic fixture.
        panel = pd.DataFrame(
            {
                "source_column_index": np.arange(PANEL_SIZE),
                "scgpt_token": [f"G{i}" for i in range(PANEL_SIZE)],
            }
        )
        control_fixture = {
            ("D1", "S1", f"N{i}"): rng.integers(1, 30, size=PANEL_SIZE).astype(float)
            for i in range(12)
        }
        library_fixture = {key: float(value.sum()) for key, value in control_fixture.items()}
        edges, index = build_train_ntc_coexpression(
            control_fixture, library_fixture, panel, k=10, threshold=0.4
        )
        if len(index) != 12 or set(panel.scgpt_token) - set(edges.target):
            raise AssertionError("Train NTC coexpression self-test failed")
    print("E168 asset-builder self-test: PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pretruth", "postgate"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gate-snapshot", type=Path)
    parser.add_argument("--gate-commit")
    parser.add_argument("--branch")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.batch_size > 512:
        parser.error("--batch-size must be between 1 and 512")
    if not args.self_test and args.stage is None:
        parser.error("--stage is required unless --self-test is used")
    if args.stage == "postgate" and (not args.gate_snapshot or not args.gate_commit):
        parser.error("postgate requires --gate-snapshot and --gate-commit")
    return args


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.stage == "pretruth":
        output = build_pretruth(args.batch_size)
    else:
        branch = args.branch or git_text("rev-parse", "--abbrev-ref", "HEAD")
        if branch == "HEAD":
            raise IntegrityError("Postgate requires a named branch for dual-remote verification")
        output = build_postgate(
            batch_size=args.batch_size,
            snapshot_path=args.gate_snapshot,
            gate_commit=args.gate_commit,
            branch=branch,
        )
    print(json.dumps({"status": "PASS", "stage": args.stage, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
