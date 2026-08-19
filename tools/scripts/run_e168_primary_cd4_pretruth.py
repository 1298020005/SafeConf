#!/usr/bin/env python3
"""E168 F2: train two model families and freeze a truth-blind RIAG gate.

The runner accepts only the isolated ``F2_pretruth`` asset bundle.  In
particular, prediction graphs never contain ``y`` and the final donor's
targeting expression is neither present in the bundle nor opened here.
Post-gate truth evaluation belongs to a different program and process.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
import platform
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Iterable
import uuid

import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
EXPERIMENT_ID = "E168_primary_human_cd4_fresh_confirmation"
SNAPSHOT_SCHEMA = "safeconf_e168_pretruth_gate_snapshot_v1"
PRETRUTH_GATE_STAGE = "F2_PRETRUTH_GATE"
PRETRUTH_ASSET_STAGE = "F2_PRETRUTH_ISOLATED_ASSET_BUILD"
EXPECTED_ASSET_DIR_NAME = "F2_pretruth"
PRETRUTH_REPORT_NAME = "E168_PRETRUTH_REPORT.md"
PRETRUTH_REPORT_TITLE = "E168 pretruth gate"
G4_SEED_NAMESPACE = "E168_G4"
# The original E168/E170 gate compares matched single-seed model pairs.  Fresh
# protocols may explicitly opt into leave-one-seed-out family means so that G4
# estimates the same family-averaged object used by the deployed score.
G4_RISK_MODE = "single_seed_pair"
OUT = ROOT / "docs/实验结果/E168_primary_human_cd4_fresh_confirmation_20260716"
DEFAULT_ASSETS = Path(
    "/home/yyf/data/safeconf_external/primary_cd4_perturbseq_2025/isolated/F2_pretruth"
)
RELEASE = OUT / "pretruth_release"
STAGING = OUT / ".pretruth_release.staging"
TASK_MANIFEST = OUT / "manifests/E168_TASK_MANIFEST.csv"
SELECTED_TARGETS = OUT / "manifests/E168_SELECTED_TARGETS.csv"
DONOR_ROLES = OUT / "manifests/E168_DONOR_STATE_ROLES.csv"
MODEL_LOCK = OUT / "MODEL_INPUT_LOCK.json"
SOURCE_LOCK = OUT / "SOURCE_LOCK.json"
ANALYSIS_PLAN = OUT / "PREREG_ANALYSIS_PLAN.md"
PROTOCOL = (
    ROOT
    / "code/20260426_154505_perturb_transport_final_push"
    / "safetrans_confidence/scoring/protocol_v0_2.py"
)
E65_SCRIPT = ROOT / "tools/scripts/run_e65_scgpt_formal_fixed_panel.py"
ASSET_BUILDER = ROOT / "tools/scripts/build_e168_primary_cd4_isolated_assets.py"
SCGPT_CHECKPOINT = Path(
    "/home/yyf/archive/code/20260519_0958_home_cleanup/moved_top_level/"
    "codex_scgpt_attnres_workspace/checkpoints/whole-human"
)
GO_FILE = Path("/home/yyf/data/gears_formal_baselines_v2/frangieh_local_atlas/go.csv")

STATES = ("Rest", "Stim8hr", "Stim48hr")
SEEDS = (3407, 3408, 3409)
N_GENES = 512
N_TARGETS = 200
N_SEEN = 160
N_UNSEEN = 40
N_TRAIN_REFERENCE = 960
N_VALIDATION_QUERY = 600
N_TEST_QUERY = 600
SCORE_TOL = 1e-6
PREDICTION_TOL = 1e-6
MAGNITUDE_TOL = 1e-6
G4_BOOTSTRAPS = 2000
RC_COVERAGES = np.linspace(0.20, 1.00, 17)
EXPECTED_ACCESS_PHASE_COUNTS = {
    "PRETRUTH_CONTROL_X": 11018,
    "PRETRUTH_TRAIN_X": 1920,
    "PRETRUTH_VALIDATION_X": 960,
}

ASSET_ALLOWLIST = {
    "GENE_PANEL.csv",
    "CONTROL_PROFILES.npz",
    "SEEN_TARGET_EFFECTS.npz",
    "PRETRUTH_TASKS.csv",
    "PRETRUTH_GUIDE_EFFECT_INDEX.csv",
    "TRAIN_NTC_COEXPRESSION_EDGES.csv",
    "TRAIN_NTC_COEXPRESSION_PROFILE_INDEX.csv",
    "ROW_ACCESS_AUDIT.csv",
    "ACCESS_ATTESTATION.json",
    "MANIFEST.sha256",
}
FORBIDDEN_SUFFIXES = {".h5", ".h5ad", ".h5mu", ".loom", ".zarr"}


class IntegrityFailure(RuntimeError):
    """A fail-closed contract or gate violation."""


@dataclass
class Assets:
    panel: pd.DataFrame
    tasks: pd.DataFrame
    controls: dict[str, np.ndarray]
    seen_effects: dict[str, np.ndarray]
    coexpression: pd.DataFrame
    attestation: dict[str, Any]
    manifest_sha256: str
    input_hashes: list[dict[str, Any]]


def now() -> str:
    return pd.Timestamp.now(tz="Asia/Shanghai").isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: str) -> int:
    payload = "\0".join(parts).encode()
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    atomic_bytes(path, payload.encode())


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_bytes(path, frame.to_csv(index=False, float_format="%.17g").encode())


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def set_seed(value: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(value)
    random.seed(value)
    np.random.seed(value)
    try:
        import torch

        torch.manual_seed(value)
        torch.cuda.manual_seed_all(value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def import_script(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise IntegrityFailure(f"cannot import frozen module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def git_text(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def verify_dual_remote_contains_head(head: str) -> tuple[str, dict[str, str]]:
    branch = git_text("rev-parse", "--abbrev-ref", "HEAD")
    if branch == "HEAD":
        raise IntegrityFailure("formal E168 pretruth run requires a named Git branch")
    remote_heads: dict[str, str] = {}
    for remote in ("origin", "github"):
        result = subprocess.run(
            [
                "git", "fetch", "--quiet", remote,
                f"refs/heads/{branch}:refs/remotes/{remote}/{branch}",
            ],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode:
            raise IntegrityFailure(
                f"cannot verify pretruth code freeze on {remote}: "
                f"{result.stderr.decode(errors='replace').strip()}"
            )
        remote_head = git_text("rev-parse", f"refs/remotes/{remote}/{branch}")
        if subprocess.run(
            ["git", "merge-base", "--is-ancestor", head, remote_head],
            cwd=ROOT,
            check=False,
        ).returncode:
            raise IntegrityFailure(f"pretruth code HEAD {head} is absent from {remote}/{branch}")
        remote_heads[remote] = remote_head
    return branch, remote_heads


def strict_bool(series: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise IntegrityFailure(f"{name} contains a non-boolean value")
    return normalized.eq("true")


def require_committed(path: Path, head: str) -> dict[str, Any]:
    relative = path.relative_to(ROOT).as_posix()
    if not path.is_file() or path.is_symlink():
        raise IntegrityFailure(f"missing or symlinked frozen file: {relative}")
    payload = path.read_bytes()
    try:
        committed = subprocess.check_output(["git", "show", f"{head}:{relative}"], cwd=ROOT)
    except subprocess.CalledProcessError as exc:
        raise IntegrityFailure(f"required file not committed: {relative}") from exc
    if hashlib.sha256(payload).digest() != hashlib.sha256(committed).digest():
        raise IntegrityFailure(f"working file differs from HEAD: {relative}")
    return {"path": relative, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def parse_sha_manifest(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise IntegrityFailure("invalid F2 MANIFEST.sha256 line")
        name = parts[1].lstrip("* ").strip()
        if "/" in name or name in result:
            raise IntegrityFailure("unsafe or duplicate F2 manifest name")
        result[name] = parts[0]
    expected = ASSET_ALLOWLIST - {"MANIFEST.sha256"}
    if set(result) != expected:
        raise IntegrityFailure(
            f"F2 manifest allowlist mismatch: missing={sorted(expected-set(result))}, "
            f"extra={sorted(set(result)-expected)}"
        )
    return result


def load_npz_vectors(path: Path) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as archive:
        for key in archive.files:
            vector = np.asarray(archive[key], dtype=np.float32)
            if vector.shape != (N_GENES,) or not np.isfinite(vector).all():
                raise IntegrityFailure(f"invalid vector in {path.name}: {key}/{vector.shape}")
            result[str(key)] = vector
    return result


def validate_asset_directory(asset_root: Path) -> tuple[dict[str, str], list[dict[str, Any]]]:
    if asset_root.is_symlink():
        raise IntegrityFailure("F2 asset root must not be a symbolic link")
    root = asset_root.resolve(strict=True)
    if root.is_symlink() or root.name != EXPECTED_ASSET_DIR_NAME:
        raise IntegrityFailure("runner accepts only a real directory named F2_pretruth")
    observed = {p.name for p in root.iterdir() if p.is_file()}
    directories = [p for p in root.iterdir() if p.is_dir()]
    if directories or observed != ASSET_ALLOWLIST:
        raise IntegrityFailure(
            f"F2 exact allowlist failed: files={sorted(observed)}, dirs={[p.name for p in directories]}"
        )
    for path in root.rglob("*"):
        lowered = "/".join(part.lower() for part in path.parts)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or "f3_postgate" in lowered:
            raise IntegrityFailure(f"raw/postgate path entered F2 bundle: {path}")
    manifest = parse_sha_manifest(root / "MANIFEST.sha256")
    hashes = []
    for name, expected in sorted(manifest.items()):
        path = root / name
        observed_sha = sha256_file(path)
        if observed_sha != expected:
            raise IntegrityFailure(f"F2 asset hash mismatch: {name}")
        hashes.append({"path": name, "bytes": path.stat().st_size, "sha256": observed_sha})
    hashes.append(
        {
            "path": "MANIFEST.sha256",
            "bytes": (root / "MANIFEST.sha256").stat().st_size,
            "sha256": sha256_file(root / "MANIFEST.sha256"),
        }
    )
    return manifest, hashes


def load_assets(asset_root: Path) -> Assets:
    manifest, hashes = validate_asset_directory(asset_root)
    root = asset_root.resolve()
    panel = pd.read_csv(root / "GENE_PANEL.csv", keep_default_na=False)
    required_panel = {
        "panel_index", "ensembl_id", "gene_name", "scgpt_token", "panel_role",
        "train_ntc_mean_expression",
    }
    if not required_panel.issubset(panel.columns) or len(panel) != N_GENES:
        raise IntegrityFailure("E168 panel schema/count failed")
    panel = panel.sort_values("panel_index").reset_index(drop=True)
    if not np.array_equal(panel.panel_index.to_numpy(int), np.arange(N_GENES)):
        raise IntegrityFailure("E168 panel indexes are not 0..511")
    if panel.ensembl_id.astype(str).duplicated().any() or panel.scgpt_token.astype(str).duplicated().any():
        raise IntegrityFailure("E168 panel ID/token uniqueness failed")
    if not np.isfinite(panel.train_ntc_mean_expression.to_numpy(float)).all():
        raise IntegrityFailure("non-finite train-only panel ranking statistic")
    if panel.panel_role.value_counts().to_dict() != {
        "REGISTERED_TARGET": N_TARGETS,
        "TRAIN_NTC_HIGH_EXPRESSION": N_GENES - N_TARGETS,
    }:
        raise IntegrityFailure("E168 panel role counts are not frozen 200+312")
    if not panel.iloc[:N_TARGETS].panel_role.eq("REGISTERED_TARGET").all():
        raise IntegrityFailure("registered target genes are not the first 200 panel entries")

    frozen = pd.read_csv(TASK_MANIFEST, keep_default_na=False)
    tasks = pd.read_csv(root / "PRETRUTH_TASKS.csv", keep_default_na=False)
    shared = list(frozen.columns)
    if not set(shared).issubset(tasks.columns) or len(tasks) != 4 * 3 * N_TARGETS:
        raise IntegrityFailure("PRETRUTH_TASKS schema/count failed")
    left = frozen[shared].sort_values("task_id").reset_index(drop=True).astype(str)
    right = tasks[shared].sort_values("task_id").reset_index(drop=True).astype(str)
    if not left.equals(right):
        raise IntegrityFailure("F2 tasks differ from committed expression-blind task manifest")
    if tasks.task_id.duplicated().any():
        raise IntegrityFailure("duplicate E168 task_id")
    if tasks.donor_role.value_counts().to_dict() != {
        "train": 1200, "validation": 600, "test": 600
    }:
        raise IntegrityFailure("E168 task donor-role counts changed")
    if tasks.target_stratum.value_counts().to_dict() != {
        "DONOR_UNSEEN_ONLY": 1920, "COLUMN_UNSEEN": 480
    }:
        raise IntegrityFailure("E168 task target-stratum counts changed")

    controls = load_npz_vectors(root / "CONTROL_PROFILES.npz")
    donor = pd.read_csv(DONOR_ROLES).drop_duplicates("donor_id")
    expected_controls = {
        f"{row.donor_id}::{state}" for row in donor.itertuples(index=False) for state in STATES
    }
    if set(controls) != expected_controls or len(controls) != 12:
        raise IntegrityFailure("F2 control keys must be exactly 4 donors x 3 states")

    seen_effects = load_npz_vectors(root / "SEEN_TARGET_EFFECTS.npz")
    expected_effects = set(
        tasks.loc[
            tasks.target_stratum.eq("DONOR_UNSEEN_ONLY")
            & tasks.donor_role.isin(["train", "validation"]),
            "task_id",
        ].astype(str)
    )
    forbidden_ids = set(
        tasks.loc[
            tasks.donor_role.eq("test") | tasks.target_stratum.eq("COLUMN_UNSEEN"), "task_id"
        ].astype(str)
    )
    if set(seen_effects) != expected_effects or set(seen_effects) & forbidden_ids:
        raise IntegrityFailure(
            f"F2 effects boundary failed: observed={len(seen_effects)}, expected={len(expected_effects)}"
        )
    if len(seen_effects) != 1440:
        raise IntegrityFailure("F2 effects must contain 960 train + 480 validation seen tasks")

    coexpression = pd.read_csv(root / "TRAIN_NTC_COEXPRESSION_EDGES.csv")
    required_edges = {"source", "target", "importance"}
    if not required_edges.issubset(coexpression.columns) or coexpression.empty:
        raise IntegrityFailure("train-NTC coexpression edge schema failed")
    tokens = set(panel.scgpt_token.astype(str))
    if not set(coexpression.source.astype(str)).issubset(tokens) or not set(
        coexpression.target.astype(str)
    ).issubset(tokens):
        raise IntegrityFailure("coexpression edge contains gene outside frozen panel")
    if not np.isfinite(coexpression.importance.to_numpy(float)).all():
        raise IntegrityFailure("non-finite coexpression weight")
    if (coexpression.importance.to_numpy(float) < 0).any():
        raise IntegrityFailure("coexpression contains a negative absolute-correlation weight")

    profile_index = pd.read_csv(
        root / "TRAIN_NTC_COEXPRESSION_PROFILE_INDEX.csv", keep_default_na=False
    )
    required_profile = {
        "profile_index", "donor_id", "culture_condition", "ntc_guide_id",
        "raw_library_sum", "used_for_train_only_coexpression",
    }
    if not required_profile.issubset(profile_index.columns) or profile_index.empty:
        raise IntegrityFailure("train-NTC coexpression profile index schema failed")
    train_donors = set(tasks.loc[tasks.donor_role.eq("train"), "donor_id"].astype(str))
    if not set(profile_index.donor_id.astype(str)).issubset(train_donors):
        raise IntegrityFailure("validation/test donor entered train-only coexpression profiles")
    if not strict_bool(
        profile_index.used_for_train_only_coexpression,
        "used_for_train_only_coexpression",
    ).all():
        raise IntegrityFailure("a coexpression profile lacks the train-only marker")
    if not np.isfinite(profile_index.raw_library_sum.to_numpy(float)).all() or (
        profile_index.raw_library_sum.to_numpy(float) <= 0
    ).any():
        raise IntegrityFailure("coexpression profile has an invalid raw library sum")

    access = pd.read_csv(root / "ROW_ACCESS_AUDIT.csv", keep_default_na=False)
    required_access = {
        "metadata_row_index", "x_access_phase", "logical_x_row_read_count",
        "asset_stage", "purpose",
    }
    if not required_access.issubset(access.columns):
        raise IntegrityFailure("F2 row-access audit schema failed")
    if len(access) != sum(EXPECTED_ACCESS_PHASE_COUNTS.values()):
        raise IntegrityFailure("F2 row-access audit count changed")
    if access.metadata_row_index.astype(int).duplicated().any():
        raise IntegrityFailure("F2 row-access audit contains a duplicate source row")
    if access.x_access_phase.value_counts().to_dict() != EXPECTED_ACCESS_PHASE_COUNTS:
        raise IntegrityFailure("F2 row-access phase counts changed")
    if set(access.asset_stage.astype(str)) != {"F2_PRETRUTH"}:
        raise IntegrityFailure("F2 row-access audit contains another asset stage")
    if not access.logical_x_row_read_count.astype(int).eq(1).all():
        raise IntegrityFailure("F2 row-access audit is not exactly-once")

    guide_index = pd.read_csv(
        root / "PRETRUTH_GUIDE_EFFECT_INDEX.csv", keep_default_na=False
    )
    if len(guide_index) != 2880:
        raise IntegrityFailure("F2 guide-effect audit must contain 2,880 guide/context rows")
    if set(guide_index.target_stratum.astype(str)) != {"DONOR_UNSEEN_ONLY"}:
        raise IntegrityFailure("column-unseen guide truth entered the F2 guide audit")
    if set(guide_index.x_access_phase.astype(str)) != {
        "PRETRUTH_TRAIN_X", "PRETRUTH_VALIDATION_X"
    }:
        raise IntegrityFailure("F2 guide-effect audit contains a forbidden access phase")

    attestation = json.loads((root / "ACCESS_ATTESTATION.json").read_text())
    required_zero = {
        "test_targeting_x_values_read": 0,
        "pretruth_test_targeting_x_values_read": 0,
        "forbidden_column_unseen_x_values_read": 0,
    }
    for key, expected in required_zero.items():
        if attestation.get(key) != expected:
            raise IntegrityFailure(f"F2 access attestation failed: {key}")
    if (
        attestation.get("status") != "PASS"
        or attestation.get("stage") != PRETRUTH_ASSET_STAGE
        or attestation.get("experiment") != EXPERIMENT_ID
    ):
        raise IntegrityFailure("F2 access attestation stage failed")
    if attestation.get("logical_x_rows_read") != sum(EXPECTED_ACCESS_PHASE_COUNTS.values()):
        raise IntegrityFailure("F2 access attestation row count failed")
    if attestation.get("logical_x_rows_read_by_phase") != EXPECTED_ACCESS_PHASE_COUNTS:
        raise IntegrityFailure("F2 access attestation phase counts failed")
    if attestation.get("builder_sha256") != sha256_file(ASSET_BUILDER):
        raise IntegrityFailure("F2 attestation is not bound to the current frozen asset builder")
    asset_head = str(attestation.get("current_git_head", ""))
    if subprocess.run(
        ["git", "cat-file", "-e", f"{asset_head}^{{commit}}"],
        cwd=ROOT,
        check=False,
    ).returncode or subprocess.run(
        ["git", "merge-base", "--is-ancestor", asset_head, git_head()],
        cwd=ROOT,
        check=False,
    ).returncode:
        raise IntegrityFailure("F2 asset-build Git head is absent from current history")
    if attestation.get("source_official_crc64nvme_base64") != "E2slkXBEb2c=":
        raise IntegrityFailure("F2 attestation lacks the official source CRC64NVME")
    source_sha = str(attestation.get("source_full_sha256", ""))
    if len(source_sha) != 64 or any(character not in "0123456789abcdef" for character in source_sha):
        raise IntegrityFailure("F2 attestation lacks full source SHA-256")

    return Assets(
        panel=panel,
        tasks=tasks,
        controls=controls,
        seen_effects=seen_effects,
        coexpression=coexpression,
        attestation=attestation,
        manifest_sha256=sha256_file(root / "MANIFEST.sha256"),
        input_hashes=hashes,
    )


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-12 else 0.0


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def quantize(values: np.ndarray, tolerance: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise IntegrityFailure("cannot quantize non-finite values")
    return np.rint(values / tolerance).astype(np.int64)


def score_gate(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, float)
    finite = bool(np.isfinite(values).all())
    if not finite:
        return {"all_finite": False, "n_levels": 0, "std": float("nan"), "passed": False}
    levels = np.unique(quantize(values, SCORE_TOL))
    std = float(np.std(values, ddof=0))
    return {
        "all_finite": True,
        "n_levels": int(len(levels)),
        "std": std,
        "passed": bool(len(levels) >= 2 and std > SCORE_TOL),
    }


def predictor_gate(matrix: np.ndarray) -> dict[str, Any]:
    values = np.asarray(matrix, float)
    finite = bool(values.ndim == 2 and np.isfinite(values).all())
    if not finite:
        return {
            "all_finite": False, "n_unique_vectors": 0, "max_coordinate_std": float("nan"),
            "passed": False,
        }
    encoded = np.ascontiguousarray(quantize(values, PREDICTION_TOL))
    fingerprints = Counter(hashlib.sha256(row.tobytes()).digest() for row in encoded)
    max_std = float(np.max(np.std(values, axis=0, ddof=0)))
    return {
        "all_finite": True,
        "n_unique_vectors": int(len(fingerprints)),
        "max_repeat_fraction": float(max(fingerprints.values()) / len(values)),
        "max_coordinate_std": max_std,
        "passed": bool(len(fingerprints) >= 2 and max_std > PREDICTION_TOL),
    }


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 4 or np.unique(a[keep]).size < 2 or np.unique(b[keep]).size < 2:
        return float("nan")
    return float(np.corrcoef(rankdata(a[keep], method="average"), rankdata(b[keep], method="average"))[0, 1])


def kendall_w(matrix: np.ndarray) -> float:
    values = np.asarray(matrix, float)
    ranks = np.stack([rankdata(row, method="average") for row in values])
    n, m = ranks.shape[1], ranks.shape[0]
    rank_sums = ranks.sum(axis=0)
    numerator = 12.0 * float(np.sum((rank_sums - m * (n + 1) / 2.0) ** 2))
    tie_correction = 0.0
    for row in values:
        counts = np.unique(row, return_counts=True)[1]
        tie_correction += float(np.sum(counts**3 - counts))
    denominator = m * m * (n**3 - n) - m * tie_correction
    return numerator / denominator if denominator > 0 else float("nan")


def g4_stability(seed_risks: np.ndarray, seed: int, n_boot: int = G4_BOOTSTRAPS) -> dict[str, Any]:
    risks = np.asarray(seed_risks, float)
    if risks.shape[0] != 3 or risks.ndim != 2:
        raise IntegrityFailure("G4 requires exactly three aligned seed-risk vectors")
    pairs = [(0, 1), (0, 2), (1, 2)]
    correlations = np.asarray([spearman(risks[a], risks[b]) for a, b in pairs], float)
    median = float(np.nanmedian(correlations))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        take = rng.integers(0, risks.shape[1], risks.shape[1])
        values = [spearman(risks[a, take], risks[b, take]) for a, b in pairs]
        if np.isfinite(values).all():
            boot.append(float(np.median(values)))
    lower = float(np.quantile(boot, 0.025)) if boot else float("nan")
    upper = float(np.quantile(boot, 0.975)) if boot else float("nan")
    return {
        "spearman_3407_3408": correlations[0],
        "spearman_3407_3409": correlations[1],
        "spearman_3408_3409": correlations[2],
        "median_pairwise_spearman": median,
        "kendall_w": kendall_w(risks),
        "bootstrap_valid": len(boot),
        "bootstrap_ci95_lower": lower,
        "bootstrap_ci95_upper": upper,
        "passed": bool(math.isfinite(median) and median >= 0.5 and math.isfinite(lower) and lower > 0),
    }


def weak_order_identical(a: np.ndarray, b: np.ndarray) -> bool:
    qa, qb = quantize(a, SCORE_TOL), quantize(b, MAGNITUDE_TOL)
    return bool(np.array_equal(rankdata(qa, method="average"), rankdata(qb, method="average")))


def boundary_record(score: np.ndarray, coverage: float, direction: str) -> dict[str, Any]:
    labels = quantize(score, SCORE_TOL)
    k = max(1, int(math.ceil(coverage * len(labels))))
    if direction == "lowest_risk_accept":
        threshold = np.sort(labels)[k - 1]
        strict = int(np.sum(labels < threshold))
    elif direction == "highest_risk_review":
        threshold = np.sort(labels)[::-1][k - 1]
        strict = int(np.sum(labels > threshold))
    else:
        raise ValueError(direction)
    tied = int(np.sum(labels == threshold))
    slots = k - strict
    return {
        "coverage": float(coverage), "direction": direction, "selected_k": k,
        "strictly_selected": strict, "boundary_tie_size": tied,
        "boundary_slots_needed": slots,
        "boundary_status": "EXACT_SET" if tied == slots else "TIEBREAK_REQUIRED",
    }


def query_batch_size(batch: Any) -> int:
    """The sole batch-size source for truth-free model forward passes."""
    if not hasattr(batch, "pert"):
        raise IntegrityFailure("query batch lacks task identifiers")
    if hasattr(batch, "y") and getattr(batch, "y") is not None:
        raise IntegrityFailure("query-only graph unexpectedly contains y")
    return len(batch.pert)


def synthetic_tests() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(test_id: str, passed: bool, observed: str) -> None:
        rows.append({"test_id": test_id, "passed": bool(passed), "observed": observed})

    constant = np.ones(100)
    variable = np.linspace(0, 1, 100)
    add("S1_constant_score_fails", not score_gate(constant)["passed"], str(score_gate(constant)))
    add("S2_variable_score_passes", score_gate(variable)["passed"], str(score_gate(variable)))
    jitter = np.arange(100) * 1e-12
    add("S3_epsilon_jitter_fails", not score_gate(jitter)["passed"], str(score_gate(jitter)))
    collapsed = np.ones((100, 8))
    diverse = np.column_stack([variable, variable**2])
    add("S4_constant_predictor_fails", not predictor_gate(collapsed)["passed"], str(predictor_gate(collapsed)))
    add("S5_task_dependent_predictor_passes", predictor_gate(diverse)["passed"], str(predictor_gate(diverse)))

    class Query:
        pert = ["a", "b", "c"]

    query = Query()
    add("S6_query_batch_without_y", query_batch_size(query) == 3 and not hasattr(query, "y"), "batch=3")
    pooled = np.concatenate([np.zeros(50), np.ones(50)])
    add(
        "S7_cross_batch_pooling_detected",
        score_gate(pooled)["passed"] and not score_gate(pooled[:50])["passed"] and not score_gate(pooled[50:])["passed"],
        "pooled_pass/state_fail",
    )
    clone = 3 + 2 * variable
    add("S8_operational_clone_detected", weak_order_identical(variable, clone), "clone=True")
    stable = np.stack([variable, variable + 0.001 * np.sin(np.arange(100)), variable + 0.001 * np.cos(np.arange(100))])
    stable_result = g4_stability(stable, 17, n_boot=250)
    add("S9_stable_seed_ranks_pass", stable_result["passed"], str(stable_result))
    rng = np.random.default_rng(18)
    unstable = np.stack([rng.permutation(variable) for _ in range(3)])
    unstable_result = g4_stability(unstable, 19, n_boot=250)
    add("S10_unstable_seed_ranks_fail", not unstable_result["passed"], str(unstable_result))
    return pd.DataFrame(rows)


def build_graphs(assets: Assets) -> tuple[dict[str, list[Any]], list[Any], pd.DataFrame]:
    import torch
    from torch_geometric.data import Data

    target_position = {
        row.ensembl_id: int(row.panel_index)
        for row in assets.panel.itertuples(index=False)
        if str(row.ensembl_id) in set(assets.tasks.perturbed_gene_id.astype(str))
    }
    if len(target_position) != N_TARGETS:
        raise IntegrityFailure(f"all 200 targets must occur in 512 panel, observed {len(target_position)}")

    supervised: dict[str, list[Any]] = {"train": [], "validation": []}
    query: list[Any] = []
    audit = []
    for row in assets.tasks.itertuples(index=False):
        task_id = str(row.task_id)
        control_key = f"{row.donor_id}::{row.culture_condition}"
        basal = assets.controls[control_key]
        flag = np.zeros(N_GENES, np.float32)
        flag[target_position[str(row.perturbed_gene_id)]] = 1.0
        x = torch.from_numpy(np.stack([basal, flag], axis=1))

        should_query = (
            (row.donor_role == "train" and row.target_stratum == "DONOR_UNSEEN_ONLY")
            or row.donor_role in {"validation", "test"}
        )
        if should_query:
            graph = Data(
                x=x, pert=task_id, donor_id=str(row.donor_id),
                culture_condition=str(row.culture_condition),
                perturbed_gene_id=str(row.perturbed_gene_id),
                target_stratum=str(row.target_stratum), split=str(row.split),
            )
            if getattr(graph, "y", None) is not None:
                raise IntegrityFailure("query graph construction attached y")
            query.append(graph)
            audit.append({"task_id": task_id, "graph_role": "query", "contains_y": False})

        if task_id in assets.seen_effects:
            target = basal + assets.seen_effects[task_id]
            graph = Data(
                x=x, y=torch.from_numpy(target).unsqueeze(0), pert=task_id,
                donor_id=str(row.donor_id), culture_condition=str(row.culture_condition),
                perturbed_gene_id=str(row.perturbed_gene_id), target_stratum=str(row.target_stratum),
                split=str(row.split),
            )
            supervised[str(row.donor_role)].append(graph)
            audit.append({"task_id": task_id, "graph_role": f"supervised_{row.donor_role}", "contains_y": True})

    if len(supervised["train"]) != 960 or len(supervised["validation"]) != 480:
        raise IntegrityFailure("supervised graph counts must be train=960, validation=480")
    if len(query) != N_TRAIN_REFERENCE + N_VALIDATION_QUERY + N_TEST_QUERY:
        raise IntegrityFailure("query graph count must be 2160")
    test_ids = set(assets.tasks.loc[assets.tasks.donor_role.eq("test"), "task_id"].astype(str))
    query_test = [graph for graph in query if str(graph.pert) in test_ids]
    if len(query_test) != 600 or any(
        getattr(graph, "y", None) is not None for graph in query_test
    ):
        raise IntegrityFailure("test query graph isolation failed")
    return supervised, query, pd.DataFrame(audit)


def scgpt_query_forward(model: Any, batch: Any, gene_ids: np.ndarray, device: Any, amp: bool, e65: Any) -> Any:
    import torch

    batch_size = query_batch_size(batch)
    batch = batch.to(device)
    n_genes = len(gene_ids)
    values = batch.x[:, 0].view(batch_size, n_genes)
    flags = batch.x[:, 1].long().view(batch_size, n_genes)
    raw_index = torch.arange(n_genes, device=device, dtype=torch.long)
    mapped = e65.map_raw_id_to_vocab_id(raw_index, gene_ids).repeat(batch_size, 1)
    mask = torch.zeros_like(values, dtype=torch.bool, device=device)
    with torch.cuda.amp.autocast(enabled=amp):
        return model(
            mapped, values, flags, src_key_padding_mask=mask,
            CLS=False, CCE=False, MVC=False, ECS=False, do_sample=False,
        )["mlm_output"]


def supervised_scgpt_forward(model: Any, batch: Any, gene_ids: np.ndarray, device: Any, amp: bool, e65: Any) -> tuple[Any, Any]:
    import torch

    batch_size = len(batch.pert)
    batch = batch.to(device)
    n_genes = len(gene_ids)
    values = batch.x[:, 0].view(batch_size, n_genes)
    flags = batch.x[:, 1].long().view(batch_size, n_genes)
    raw_index = torch.arange(n_genes, device=device, dtype=torch.long)
    mapped = e65.map_raw_id_to_vocab_id(raw_index, gene_ids).repeat(batch_size, 1)
    mask = torch.zeros_like(values, dtype=torch.bool, device=device)
    with torch.cuda.amp.autocast(enabled=amp):
        prediction = model(mapped, values, flags, src_key_padding_mask=mask, CLS=False, CCE=False, MVC=False, ECS=False, do_sample=False)["mlm_output"]
    return prediction, batch.y


def train_scgpt(seed_value: int, supervised: dict[str, list[Any]], query: list[Any], genes: list[str], device: Any) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, Any]]:
    import torch
    from torch_geometric.loader import DataLoader

    set_seed(seed_value)
    e65 = import_script(f"e65_for_e168_{seed_value}", E65_SCRIPT)
    model, _, meta = e65.load_model(device)
    gene_ids = e65.make_gene_ids(genes, meta["vocab"])
    train_loader = DataLoader(supervised["train"], 16, shuffle=True, generator=torch.Generator().manual_seed(seed_value))
    val_loader = DataLoader(supervised["validation"], 16, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    best_loss, best_epoch, stale, best_state = float("inf"), 0, 0, None
    history = []
    for epoch in range(1, 11):
        model.train(); train_losses = []
        for batch in train_loader:
            prediction, target = supervised_scgpt_forward(model, batch, gene_ids, device, amp, e65)
            loss = torch.mean((prediction - target) ** 2)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            train_losses.append(float(loss.detach().cpu()))
        model.eval(); val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                prediction, target = supervised_scgpt_forward(model, batch, gene_ids, device, amp, e65)
                val_losses.append(float(torch.mean((prediction - target) ** 2).detach().cpu()))
        train_loss, val_loss = float(np.mean(train_losses)), float(np.mean(val_losses))
        history.append({"seed": seed_value, "model": "scGPT", "epoch": epoch, "train_mse": train_loss, "validation_mse": val_loss})
        if val_loss < best_loss - 1e-7:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 3:
                break
    if best_state is None:
        raise IntegrityFailure("scGPT produced no validation checkpoint")
    model.load_state_dict(best_state); model.to(device).eval()
    result = {}
    with torch.no_grad():
        for batch in DataLoader(query, 16, shuffle=False):
            output = scgpt_query_forward(model, batch, gene_ids, device, amp, e65)
            moved = batch.to(device)
            basal = moved.x[:, 0].view(len(batch.pert), N_GENES)
            for task_id, prediction, baseline in zip(batch.pert, output.detach().cpu().numpy(), basal.detach().cpu().numpy()):
                result[str(task_id)] = np.asarray(prediction - baseline, np.float32)
    return result, pd.DataFrame(history), {"best_epoch": best_epoch, "best_validation_mse": best_loss, "pretrained_tensors": meta["matched_pretrained_parameter_tensors"]}


def edge_tensors(frame: pd.DataFrame, genes: list[str]) -> tuple[Any, Any]:
    import torch

    node = {gene: index for index, gene in enumerate(genes)}
    clean = frame.loc[frame.source.isin(node) & frame.target.isin(node), ["source", "target", "importance"]].copy()
    clean = clean.drop_duplicates(["source", "target"], keep="first")
    existing = set(zip(clean.source.astype(str), clean.target.astype(str)))
    additions = [{"source": gene, "target": gene, "importance": 1.0} for gene in genes if (gene, gene) not in existing]
    if additions:
        clean = pd.concat([clean, pd.DataFrame(additions)], ignore_index=True)
    index = torch.tensor([[node[str(a)], node[str(b)]] for a, b in zip(clean.source, clean.target)], dtype=torch.long).T
    weight = torch.tensor(clean.importance.to_numpy(float), dtype=torch.float32)
    return index, weight


def go_edges(genes: list[str]) -> tuple[Any, Any, int]:
    go = pd.read_csv(GO_FILE)
    go = go.loc[go.source.isin(genes) & go.target.isin(genes)].copy()
    go = go.sort_values(["target", "importance"], ascending=[True, False]).groupby("target", as_index=False, group_keys=False).head(21)
    index, weight = edge_tensors(go, genes)
    return index, weight, len(go)


def train_gears(seed_value: int, supervised: dict[str, list[Any]], query: list[Any], genes: list[str], coexpression: pd.DataFrame, device: Any) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, Any]]:
    import torch
    from torch_geometric.loader import DataLoader
    from gears.model import GEARS_Model

    set_seed(seed_value)
    go_index, go_weight, n_go = go_edges(genes)
    co_index, co_weight = edge_tensors(coexpression, genes)
    config = {
        "hidden_size": 64, "num_go_gnn_layers": 1, "num_gene_gnn_layers": 1,
        "decoder_hidden_size": 16, "uncertainty": False,
        "G_go": go_index, "G_go_weight": go_weight,
        "G_coexpress": co_index, "G_coexpress_weight": co_weight,
        "device": str(device), "num_genes": N_GENES,
    }
    model = GEARS_Model(config).to(device)
    train_loader = DataLoader(supervised["train"], 16, shuffle=True, generator=torch.Generator().manual_seed(seed_value))
    val_loader = DataLoader(supervised["validation"], 16, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_loss, best_epoch, stale, best_state = float("inf"), 0, 0, None
    history = []
    for epoch in range(1, 41):
        model.train(); train_losses = []
        for batch in train_loader:
            batch = batch.to(device); prediction = model(batch); loss = torch.mean((prediction - batch.y) ** 2)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval(); val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device); prediction = model(batch)
                val_losses.append(float(torch.mean((prediction - batch.y) ** 2).detach().cpu()))
        train_loss, val_loss = float(np.mean(train_losses)), float(np.mean(val_losses))
        history.append({"seed": seed_value, "model": "GEARS", "epoch": epoch, "train_mse": train_loss, "validation_mse": val_loss})
        if val_loss < best_loss - 1e-7:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 6:
                break
    if best_state is None:
        raise IntegrityFailure("GEARS produced no validation checkpoint")
    model.load_state_dict(best_state); model.to(device).eval()
    result = {}
    with torch.no_grad():
        for batch in DataLoader(query, 16, shuffle=False):
            if getattr(batch, "y", None) is not None:
                raise IntegrityFailure("GEARS query batch unexpectedly contains y")
            batch_size = query_batch_size(batch)
            batch = batch.to(device); prediction = model(batch)
            basal = batch.x[:, 0].view(batch_size, N_GENES)
            for task_id, output, baseline in zip(batch.pert, prediction.detach().cpu().numpy(), basal.detach().cpu().numpy()):
                result[str(task_id)] = np.asarray(output - baseline, np.float32)
    return result, pd.DataFrame(history), {"best_epoch": best_epoch, "best_validation_mse": best_loss, "n_go_edges": n_go, "n_coexpression_edges": len(coexpression)}


def zscore_exact(values: pd.Series, reference: pd.Series) -> pd.Series:
    code_root = ROOT / "code/20260426_154505_perturb_transport_final_push"
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    from safetrans_confidence.scoring.protocol_v0_2 import zscore_by_ref

    result = zscore_by_ref(values, reference)
    if len(result) != len(values) or not np.isfinite(result.to_numpy(float)).all():
        raise IntegrityFailure("protocol_v0_2.zscore_by_ref returned invalid values")
    return result


def build_g4_seed_risks(
    arrays: dict[str, np.ndarray],
    train_mask: np.ndarray,
    z_context: pd.Series,
    z_support: pd.Series,
) -> dict[int, pd.Series]:
    """Return the three frozen G4 risk replicates for the configured estimator."""
    train_mask = np.asarray(train_mask, dtype=bool)
    if G4_RISK_MODE == "single_seed_pair":
        pairs = {
            seed_value: (
                arrays[f"scGPT_seed{seed_value}"], arrays[f"GEARS_seed{seed_value}"]
            )
            for seed_value in SEEDS
        }
    elif G4_RISK_MODE == "leave_one_seed_out_family_mean":
        pairs = {}
        for omitted_seed in SEEDS:
            retained = [seed for seed in SEEDS if seed != omitted_seed]
            sc_mean = np.mean(
                np.stack([arrays[f"scGPT_seed{seed}"] for seed in retained]), axis=0
            )
            ge_mean = np.mean(
                np.stack([arrays[f"GEARS_seed{seed}"] for seed in retained]), axis=0
            )
            pairs[omitted_seed] = (sc_mean, ge_mean)
    else:
        raise IntegrityFailure(f"unsupported G4 risk estimator: {G4_RISK_MODE}")
    risks: dict[int, pd.Series] = {}
    for seed_value, (sc_prediction, ge_prediction) in pairs.items():
        disagreement = np.sqrt(
            np.mean((sc_prediction - ge_prediction) ** 2, axis=1)
        )
        z_dis = zscore_exact(
            pd.Series(disagreement), pd.Series(disagreement[train_mask])
        )
        risks[seed_value] = -(z_context + z_support - z_dis)
    return risks


def assemble_scores(assets: Assets, predictions: dict[str, dict[str, np.ndarray]]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    task_order = [str(graph_task) for graph_task in assets.tasks.loc[
        (assets.tasks.donor_role.eq("train") & assets.tasks.target_stratum.eq("DONOR_UNSEEN_ONLY"))
        | assets.tasks.donor_role.isin(["validation", "test"]), "task_id"
    ]]
    if len(task_order) != 2160 or len(set(task_order)) != 2160:
        raise IntegrityFailure("scoring interface must contain 960 train + 600 val + 600 test")
    expected_predictors = {
        *(f"scGPT_seed{seed}" for seed in SEEDS),
        *(f"GEARS_seed{seed}" for seed in SEEDS),
    }
    if set(predictions) != expected_predictors:
        raise IntegrityFailure("formal prediction families/seeds differ from the frozen six")
    for name, mapping in predictions.items():
        if set(mapping) != set(task_order):
            raise IntegrityFailure(f"prediction task coverage failed: {name}")
        for task, vector in mapping.items():
            values = np.asarray(vector)
            if values.shape != (N_GENES,) or not np.isfinite(values).all():
                raise IntegrityFailure(f"invalid prediction vector: {name}/{task}/{values.shape}")

    arrays = {name: np.stack([mapping[task] for task in task_order]).astype(np.float32) for name, mapping in predictions.items()}
    sc_names = [f"scGPT_seed{seed}" for seed in SEEDS]
    ge_names = [f"GEARS_seed{seed}" for seed in SEEDS]
    arrays["scGPT_seed_mean"] = np.mean(np.stack([arrays[name] for name in sc_names]), axis=0).astype(np.float32)
    arrays["GEARS_seed_mean"] = np.mean(np.stack([arrays[name] for name in ge_names]), axis=0).astype(np.float32)
    arrays["ensemble_seed_family_mean"] = ((arrays["scGPT_seed_mean"] + arrays["GEARS_seed_mean"]) / 2).astype(np.float32)

    metadata = assets.tasks.set_index("task_id").loc[task_order].reset_index()
    train_controls = [
        vector for key, vector in assets.controls.items()
        if key.split("::", 1)[0] in set(metadata.loc[metadata.donor_role.eq("train"), "donor_id"].astype(str))
    ]
    if len(train_controls) != 6:
        raise IntegrityFailure("context reference requires six train donor-state NTC profiles")
    context = []
    for row in metadata.itertuples(index=False):
        query_control = assets.controls[f"{row.donor_id}::{row.culture_condition}"]
        context.append(max(cosine(query_control, reference) for reference in train_controls))
    metadata["context_similarity_max"] = context
    metadata["perturbation_support_count"] = np.where(metadata.target_stratum.eq("DONOR_UNSEEN_ONLY"), 6, 0)
    metadata["model_disagreement_rmse"] = np.sqrt(np.mean((arrays["scGPT_seed_mean"] - arrays["GEARS_seed_mean"]) ** 2, axis=1))
    metadata["predicted_magnitude"] = np.sqrt(np.mean(arrays["ensemble_seed_family_mean"] ** 2, axis=1))

    train = metadata.donor_role.eq("train")
    if int(train.sum()) != N_TRAIN_REFERENCE or not strict_bool(
        metadata.loc[train, "risk_reference_required"], "risk_reference_required"
    ).all():
        raise IntegrityFailure("protocol reference must be exactly the 960 frozen training predictions")
    z_context = zscore_exact(metadata.context_similarity_max, metadata.loc[train, "context_similarity_max"])
    z_support = zscore_exact(np.log1p(metadata.perturbation_support_count), np.log1p(metadata.loc[train, "perturbation_support_count"]))
    z_disagreement = zscore_exact(metadata.model_disagreement_rmse, metadata.loc[train, "model_disagreement_rmse"])
    metadata["z_context_train960"] = z_context
    metadata["z_log_support_train960"] = z_support
    metadata["z_disagreement_train960"] = z_disagreement
    metadata["safeconf_confidence"] = z_context + z_support - z_disagreement
    metadata["safeconf_risk"] = -metadata.safeconf_confidence

    seed_risks = build_g4_seed_risks(
        arrays, train.to_numpy(), z_context, z_support
    )
    for seed_value, seed_risk in seed_risks.items():
        metadata[f"seed_risk_{seed_value}"] = seed_risk
    metadata["test_targeting_truth_present"] = False
    metadata["true_error_rmse"] = np.nan
    return metadata, arrays


def run_riag(scores: pd.DataFrame, arrays: dict[str, np.ndarray], g1_passed: bool) -> dict[str, pd.DataFrame | bool]:
    test = scores.loc[scores.donor_role.eq("test")].copy()
    if len(test) != 600:
        raise IntegrityFailure("RIAG requires exactly 600 test queries")
    array_index = {task: index for index, task in enumerate(scores.task_id.astype(str))}
    predictors = [
        *(f"scGPT_seed{seed}" for seed in SEEDS), *(f"GEARS_seed{seed}" for seed in SEEDS),
        "scGPT_seed_mean", "GEARS_seed_mean",
    ]
    g2_rows, g3_rows, g4_rows, g5_rows, boundary_rows = [], [], [], [], []
    for state in STATES:
        state_all = test.culture_condition.eq(state)
        for stratum_name, mask in {
            "all_200": state_all,
            "seen_160": state_all & test.target_stratum.eq("DONOR_UNSEEN_ONLY"),
        }.items():
            block = test.loc[mask]
            expected = 200 if stratum_name == "all_200" else 160
            if len(block) != expected:
                raise IntegrityFailure(f"registered RIAG stratum count failed: {state}/{stratum_name}")
            g2 = score_gate(block.safeconf_risk.to_numpy(float))
            g2_rows.append({"culture_condition": state, "stratum": stratum_name, "n_tasks": len(block), **g2})
            seed_risks = np.stack([block[f"seed_risk_{seed}"].to_numpy(float) for seed in SEEDS])
            g4 = g4_stability(seed_risks, stable_seed(G4_SEED_NAMESPACE, state, stratum_name))
            g4_rows.append({"culture_condition": state, "stratum": stratum_name, "n_tasks": len(block), **g4})
            g5_rows.append({
                "culture_condition": state, "stratum": stratum_name, "n_tasks": len(block),
                "risk_magnitude_operational_weak_order_identical": weak_order_identical(
                    block.safeconf_risk.to_numpy(float), block.predicted_magnitude.to_numpy(float)
                ),
            })
            if stratum_name == "all_200":
                for coverage in RC_COVERAGES:
                    boundary_rows.append({"culture_condition": state, "stratum": stratum_name, **boundary_record(block.safeconf_risk.to_numpy(float), float(coverage), "lowest_risk_accept")})
                boundary_rows.append({"culture_condition": state, "stratum": stratum_name, **boundary_record(block.safeconf_risk.to_numpy(float), 0.20, "highest_risk_review")})

        all_block = test.loc[state_all]
        take = np.asarray([array_index[task] for task in all_block.task_id.astype(str)], int)
        for predictor in predictors:
            gate = predictor_gate(arrays[predictor][take])
            g3_rows.append({"culture_condition": state, "predictor_name": predictor, "n_tasks": len(take), **gate})

    synthetic = synthetic_tests()
    g2_frame, g3_frame, g4_frame, g5_frame = map(pd.DataFrame, [g2_rows, g3_rows, g4_rows, g5_rows])
    pretruth_pass = bool(
        g1_passed
        and g2_frame.passed.astype(bool).all()
        and g3_frame.passed.astype(bool).all()
        and g4_frame.passed.astype(bool).all()
        and synthetic.passed.astype(bool).all()
    )
    return {
        "g2": g2_frame, "g3": g3_frame, "g4": g4_frame, "g5": g5_frame,
        "boundaries": pd.DataFrame(boundary_rows), "synthetic": synthetic,
        "pretruth_pass": pretruth_pass,
    }


def formal_input_audit(
    asset_root: Path,
) -> tuple[str, str, dict[str, str], list[dict[str, Any]]]:
    head = git_head()
    branch, remote_heads = verify_dual_remote_contains_head(head)
    files = [
        RUNNER, ASSET_BUILDER, TASK_MANIFEST, SELECTED_TARGETS, DONOR_ROLES,
        MODEL_LOCK, SOURCE_LOCK, ANALYSIS_PLAN, PROTOCOL, E65_SCRIPT,
    ]
    hashes = [require_committed(path, head) for path in files]
    lock = json.loads(MODEL_LOCK.read_text())
    for relative, expected in lock["method_reference_files"].items():
        path = ROOT / relative
        observed = sha256_file(path)
        if observed != expected:
            raise IntegrityFailure(f"frozen method reference changed: {relative}")
        if not any(row["path"] == relative for row in hashes):
            hashes.append(require_committed(path, head))
    for path_text, expected in lock["scgpt_checkpoint_files"].items():
        path = Path(path_text)
        if sha256_file(path) != expected:
            raise IntegrityFailure(f"scGPT checkpoint changed: {path}")
        hashes.append({"path": str(path), "bytes": path.stat().st_size, "sha256": expected})
    go_lock = lock["gears_external_go_file"]
    if sha256_file(Path(go_lock["path"])) != go_lock["sha256"]:
        raise IntegrityFailure("GEARS GO prior changed")
    hashes.append({"path": go_lock["path"], "bytes": Path(go_lock["path"]).stat().st_size, "sha256": go_lock["sha256"]})
    return head, branch, remote_heads, hashes


def runtime_environment(device: Any) -> dict[str, Any]:
    import scipy
    import torch
    import torch_geometric

    gpu = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpu.append({
                "index": index,
                "name": properties.name,
                "total_memory_bytes": int(properties.total_memory),
                "compute_capability": [properties.major, properties.minor],
            })
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": int(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None,
        "selected_device": str(device),
        "gpu_inventory": gpu,
        "deterministic_cudnn": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def write_release(
    assets: Assets,
    scores: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    graph_audit: pd.DataFrame,
    histories: pd.DataFrame,
    model_audit: pd.DataFrame,
    riag: dict[str, Any],
    head: str,
    branch: str,
    remote_heads: dict[str, str],
    frozen_hashes: list[dict[str, Any]],
    environment: dict[str, Any],
) -> Path:
    if RELEASE.exists() or STAGING.exists():
        raise IntegrityFailure("E168 pretruth release is append-only and already exists")
    for sub in ("tables", "arrays", "reports"):
        (STAGING / sub).mkdir(parents=True, exist_ok=False)
    atomic_csv(STAGING / "tables/PRETRUTH_SCORING_INTERFACE.csv", scores)
    atomic_npz(STAGING / "arrays/PRETRUTH_PREDICTIONS.npz", arrays)
    atomic_csv(STAGING / "tables/QUERY_GRAPH_AUDIT.csv", graph_audit)
    atomic_csv(STAGING / "tables/TRAINING_HISTORY.csv", histories)
    atomic_csv(STAGING / "tables/MODEL_AUDIT.csv", model_audit)
    atomic_csv(STAGING / "tables/INPUT_HASHES.csv", pd.DataFrame(frozen_hashes + assets.input_hashes))
    atomic_csv(STAGING / "tables/G2_SCORE_CERTIFICATES.csv", riag["g2"])
    atomic_csv(STAGING / "tables/G3_PREDICTOR_CERTIFICATES.csv", riag["g3"])
    atomic_csv(STAGING / "tables/G4_SEED_STABILITY.csv", riag["g4"])
    atomic_csv(STAGING / "tables/G5_MAGNITUDE_EQUIVALENCE.csv", riag["g5"])
    atomic_csv(STAGING / "tables/COVERAGE_BOUNDARIES.csv", riag["boundaries"])
    atomic_csv(STAGING / "tables/SYNTHETIC_REGRESSION_TESTS.csv", riag["synthetic"])
    atomic_json(STAGING / "RUNTIME_ENVIRONMENT.json", environment)

    files = sorted(path for path in STAGING.rglob("*") if path.is_file())
    file_hashes = {
        path.relative_to(STAGING).as_posix(): sha256_file(path) for path in files
    }
    passed = bool(riag["pretruth_pass"])
    snapshot = {
        "schema": SNAPSHOT_SCHEMA,
        "experiment": EXPERIMENT_ID,
        "stage": PRETRUTH_GATE_STAGE,
        "status": "PASS" if passed else "FAIL",
        "all_registered_gates_passed": passed,
        "created_at": now(),
        "git_head": head,
        "git_branch": branch,
        "code_freeze_remote_heads": remote_heads,
        "runner_sha256": sha256_file(RUNNER),
        "asset_builder_sha256": sha256_file(ASSET_BUILDER),
        "source_full_sha256": assets.attestation["source_full_sha256"],
        "f2_manifest_sha256": assets.manifest_sha256,
        "test_targeting_x_values_read": 0,
        "forbidden_column_unseen_x_values_read": 0,
        "train_reference_task_count": int(scores.donor_role.eq("train").sum()),
        "validation_query_count": int(scores.donor_role.eq("validation").sum()),
        "test_query_count": int(scores.donor_role.eq("test").sum()),
        "test_query_graphs_containing_y": int((
            graph_audit.graph_role.eq("query")
            & graph_audit.task_id.isin(scores.loc[scores.donor_role.eq("test"), "task_id"])
            & graph_audit.contains_y.astype(bool)
        ).sum()) if len(graph_audit) else 0,
        "registered_g2_units": 6,
        "registered_g4_units": 6,
        "g4_risk_estimator": G4_RISK_MODE,
        "synthetic_tests_passed": int(riag["synthetic"].passed.astype(bool).sum()),
        "pretruth_files_sha256": file_hashes,
        "deployment_authorized": False,
    }
    atomic_json(STAGING / "PRETRUTH_GATE_SNAPSHOT.json", snapshot)
    report = (
        f"# {PRETRUTH_REPORT_TITLE}\n\n"
        f"状态：**{snapshot['status']}**。test donor targeting X 读取数为 0；"
        "所有模型输出都由不含 `y` 的 query graph 生成。\n\n"
        f"训练参考任务 {snapshot['train_reference_task_count']}，validation query "
        f"{snapshot['validation_query_count']}，test query {snapshot['test_query_count']}。\n\n"
        "本阶段没有真实 test error，也没有形成部署授权。只有将该 snapshot 作为不可变"
        "commit 同时推到 GitHub/Gitee 后，独立 postgate builder 才可尝试解封。\n"
    )
    atomic_bytes(STAGING / f"reports/{PRETRUTH_REPORT_NAME}", report.encode())
    os.replace(STAGING, RELEASE)
    return RELEASE / "PRETRUTH_GATE_SNAPSHOT.json"


def run_formal(asset_root: Path, device_name: str) -> dict[str, Any]:
    started = time.time()
    head, branch, remote_heads, frozen_hashes = formal_input_audit(asset_root)
    assets = load_assets(asset_root)
    supervised, query, graph_audit = build_graphs(assets)
    import torch

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise IntegrityFailure("registered CUDA device unavailable; no silent CPU substitution")
    genes = assets.panel.scgpt_token.astype(str).tolist()
    environment = runtime_environment(device)
    all_predictions: dict[str, dict[str, np.ndarray]] = {}
    histories, audits = [], []
    for seed_value in SEEDS:
        sc, history, audit = train_scgpt(seed_value, supervised, query, genes, device)
        all_predictions[f"scGPT_seed{seed_value}"] = sc
        histories.append(history); audits.append({"seed": seed_value, "model": "scGPT", **audit})
        del sc
        if device.type == "cuda": torch.cuda.empty_cache()
        ge, history, audit = train_gears(seed_value, supervised, query, genes, assets.coexpression, device)
        all_predictions[f"GEARS_seed{seed_value}"] = ge
        histories.append(history); audits.append({"seed": seed_value, "model": "GEARS", **audit})
        del ge
        if device.type == "cuda": torch.cuda.empty_cache()
    scores, arrays = assemble_scores(assets, all_predictions)
    g1 = bool(
        assets.attestation["test_targeting_x_values_read"] == 0
        and assets.attestation["forbidden_column_unseen_x_values_read"] == 0
        and len(graph_audit.loc[graph_audit.graph_role.eq("query") & graph_audit.contains_y.astype(bool)]) == 0
        and len(scores.loc[scores.donor_role.eq("train")]) == 960
        and len(scores.loc[scores.donor_role.eq("validation")]) == 600
        and len(scores.loc[scores.donor_role.eq("test")]) == 600
    )
    riag = run_riag(scores, arrays, g1)
    snapshot = write_release(
        assets, scores, arrays, graph_audit, pd.concat(histories, ignore_index=True),
        pd.DataFrame(audits), riag, head, branch, remote_heads, frozen_hashes,
        environment,
    )
    return {
        "status": "PASS" if riag["pretruth_pass"] else "FAIL",
        "snapshot": str(snapshot.relative_to(ROOT)),
        "snapshot_sha256": sha256_file(snapshot),
        "wall_seconds": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--synthetic-test-only", action="store_true")
    args = parser.parse_args()
    if args.synthetic_test_only:
        tests = synthetic_tests()
        print(tests.to_string(index=False))
        if not tests.passed.astype(bool).all() or len(tests) != 10:
            raise SystemExit(2)
        return
    result = run_formal(args.asset_root, args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
