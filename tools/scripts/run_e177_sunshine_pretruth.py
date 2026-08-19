#!/usr/bin/env python3
"""Run E177 five-seed pretruth gate on isolated F2 assets."""

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
import shutil
import subprocess
import sys
import time
from typing import Any
import uuid

import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
EXPERIMENT = "E177_sunshine_external_certificate"
OUT = ROOT / "docs/实验结果/E177_sunshine_external_certificate_20260719"
RELEASE = OUT / "pretruth_release"
STAGING = OUT / f".pretruth_release.staging.{os.getpid()}"
ASSET_ROOT = Path("/home/yyf/data/safeconf_e177_external/isolated/F2_pretruth")
ASSET_BUILDER = ROOT / "tools/scripts/build_e177_sunshine_pretruth_assets.py"
FREEZER = ROOT / "tools/scripts/freeze_e177_sunshine_external_certificate.py"
E65_SCRIPT = ROOT / "tools/scripts/run_e65_scgpt_formal_fixed_panel.py"
E107_SCRIPT = ROOT / "tools/scripts/run_e107_frangieh_context_gears.py"
MODEL_LOCK = OUT / "MODEL_INPUT_LOCK.json"
SOURCE_LOCK = OUT / "SOURCE_LOCK.json"
STAT_LOCK = OUT / "STATISTICAL_ANALYSIS_LOCK.json"
PLAN = OUT / "PREREG_ANALYSIS_PLAN.md"
TASKS = OUT / "manifests/E177_TASK_MANIFEST.csv"
TARGETS = OUT / "manifests/E177_SELECTED_TARGETS.csv"
F2_STATUS = OUT / "pretruth_assets/ASSET_BUILD_STATUS.json"
GO_FILE = Path("/home/yyf/data/gears_formal_baselines_v2/frangieh_local_atlas/go.csv")

SEEDS = (3407, 3408, 3409, 3410, 3411)
TECH_GROUPS = tuple(str(value) for value in range(1, 9))
N_GENES = 512
N_TARGETS = 144
N_TRAIN_TASKS = 432
N_VALIDATION_TASKS = 80
N_HIDDEN_TASKS = 640
SCORE_TOL = 1e-6
PREDICTION_TOL = 1e-6
G4_BOOTSTRAPS = 2000
ASSET_ALLOWLIST = {
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


class IntegrityFailure(RuntimeError):
    """A fail-closed gate or frozen-input check failed."""


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


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode())


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_bytes(path, frame.to_csv(index=False, float_format="%.17g").encode())


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("xb") as handle:
        np.savez_compressed(handle, **{key: np.asarray(value, np.float32) for key, value in sorted(arrays.items())})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def git_text(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def verify_dual_remote_contains_head(head: str) -> tuple[str, dict[str, str]]:
    branch = git_text("rev-parse", "--abbrev-ref", "HEAD")
    if branch == "HEAD":
        raise IntegrityFailure("E177 pretruth requires a named Git branch")
    remote_heads: dict[str, str] = {}
    for remote in ("origin", "github"):
        result = subprocess.run(
            ["git", "fetch", "--quiet", remote, f"refs/heads/{branch}:refs/remotes/{remote}/{branch}"],
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
        if subprocess.run(["git", "merge-base", "--is-ancestor", head, remote_head], cwd=ROOT, check=False).returncode:
            raise IntegrityFailure(f"pretruth HEAD {head} is absent from {remote}/{branch}")
        remote_heads[remote] = remote_head
    return branch, remote_heads


def require_committed(path: Path, head: str) -> dict[str, Any]:
    relative = path.relative_to(ROOT).as_posix()
    if not path.is_file() or path.is_symlink():
        raise IntegrityFailure(f"missing frozen file: {relative}")
    payload = path.read_bytes()
    try:
        committed = subprocess.check_output(["git", "show", f"{head}:{relative}"], cwd=ROOT)
    except subprocess.CalledProcessError as exc:
        raise IntegrityFailure(f"required file is not committed: {relative}") from exc
    if hashlib.sha256(payload).digest() != hashlib.sha256(committed).digest():
        raise IntegrityFailure(f"working file differs from HEAD: {relative}")
    return {"path": relative, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def formal_input_audit() -> tuple[str, str, dict[str, str], list[dict[str, Any]]]:
    head = git_text("rev-parse", "HEAD")
    branch, remote_heads = verify_dual_remote_contains_head(head)
    files = [
        RUNNER,
        ASSET_BUILDER,
        FREEZER,
        E65_SCRIPT,
        E107_SCRIPT,
        MODEL_LOCK,
        SOURCE_LOCK,
        STAT_LOCK,
        PLAN,
        TASKS,
        TARGETS,
        F2_STATUS,
    ]
    hashes = [require_committed(path, head) for path in files]
    model_lock = json.loads(MODEL_LOCK.read_text())
    for path_text, expected in model_lock["scgpt_checkpoint_files"].items():
        path = Path(path_text)
        if sha256_file(path) != expected:
            raise IntegrityFailure(f"scGPT checkpoint file changed: {path}")
        hashes.append({"path": str(path), "bytes": path.stat().st_size, "sha256": expected})
    vocab = Path(model_lock["scgpt_vocab"]["path"])
    if sha256_file(vocab) != model_lock["scgpt_vocab"]["sha256"]:
        raise IntegrityFailure("scGPT vocab changed")
    hashes.append({"path": str(vocab), "bytes": vocab.stat().st_size, "sha256": model_lock["scgpt_vocab"]["sha256"]})
    go = Path(model_lock["gears_go_prior"]["path"])
    if sha256_file(go) != model_lock["gears_go_prior"]["sha256"]:
        raise IntegrityFailure("GEARS GO prior changed")
    hashes.append({"path": str(go), "bytes": go.stat().st_size, "sha256": model_lock["gears_go_prior"]["sha256"]})
    return head, branch, remote_heads, hashes


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
        raise IntegrityFailure(f"cannot import script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_manifest(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("* ").strip()
        result[name] = digest
    expected = ASSET_ALLOWLIST - {"MANIFEST.sha256"}
    if set(result) != expected:
        raise IntegrityFailure(f"F2 asset manifest allowlist changed: {sorted(result)}")
    return result


def load_npz_vectors(path: Path, expected: int | None = None) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as archive:
        for key in archive.files:
            value = np.asarray(archive[key], dtype=np.float32)
            if value.shape != (N_GENES,) or not np.isfinite(value).all():
                raise IntegrityFailure(f"invalid vector in {path.name}:{key}/{value.shape}")
            result[str(key)] = value
    if expected is not None and len(result) != expected:
        raise IntegrityFailure(f"unexpected vector count in {path}: {len(result)} != {expected}")
    return result


def strict_bool(series: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise IntegrityFailure(f"{name} contains non-boolean values")
    return normalized.eq("true")


def load_assets(asset_root: Path) -> Assets:
    root = asset_root.resolve(strict=True)
    if root.is_symlink() or root.name != "F2_pretruth":
        raise IntegrityFailure("E177 runner accepts only the immutable F2_pretruth asset directory")
    observed = {path.name for path in root.iterdir() if path.is_file()}
    if observed != ASSET_ALLOWLIST:
        raise IntegrityFailure(f"E177 F2 allowlist failed: {sorted(observed)}")
    manifest = parse_manifest(root / "MANIFEST.sha256")
    hashes = []
    for name, expected in sorted(manifest.items()):
        path = root / name
        observed_sha = sha256_file(path)
        if observed_sha != expected:
            raise IntegrityFailure(f"F2 asset hash mismatch: {name}")
        hashes.append({"path": name, "bytes": path.stat().st_size, "sha256": observed_sha})
    hashes.append({"path": "MANIFEST.sha256", "bytes": (root / "MANIFEST.sha256").stat().st_size, "sha256": sha256_file(root / "MANIFEST.sha256")})

    panel = pd.read_csv(root / "GENE_PANEL.csv", keep_default_na=False)
    if len(panel) != N_GENES or not np.array_equal(panel.panel_index.to_numpy(int), np.arange(N_GENES)):
        raise IntegrityFailure("E177 panel schema/count failed")
    if panel.scgpt_token.astype(str).nunique() != N_GENES:
        raise IntegrityFailure("E177 panel token uniqueness failed")
    if panel.panel_role.value_counts().to_dict() != {
        "CONTROL_HIGH_EXPRESSION": N_GENES - N_TARGETS,
        "REGISTERED_TARGET": N_TARGETS,
    }:
        raise IntegrityFailure("E177 panel role counts changed")

    frozen_tasks = pd.read_csv(TASKS, keep_default_na=False)
    tasks = pd.read_csv(root / "PRETRUTH_TASKS.csv", keep_default_na=False)
    shared = list(frozen_tasks.columns)
    left = frozen_tasks[shared].sort_values("task_id").reset_index(drop=True).astype(str)
    right = tasks[shared].sort_values("task_id").reset_index(drop=True).astype(str)
    if not left.equals(right) or len(tasks) != N_TARGETS * len(TECH_GROUPS):
        raise IntegrityFailure("E177 F2 tasks differ from frozen metadata manifest")
    if tasks.target_split.value_counts().to_dict() != {
        "evaluation": 400,
        "train": 432,
        "calibration": 240,
        "validation": 80,
    }:
        raise IntegrityFailure("E177 task split counts changed")

    controls = load_npz_vectors(root / "CONTROL_PROFILES.npz", expected=len(TECH_GROUPS))
    if set(controls) != {f"G{group}" for group in TECH_GROUPS}:
        raise IntegrityFailure("E177 control profile keys changed")
    seen_effects = load_npz_vectors(root / "SEEN_TARGET_EFFECTS.npz", expected=N_TRAIN_TASKS + N_VALIDATION_TASKS)
    expected_seen = set(tasks.loc[tasks.target_split.isin(["train", "validation"]), "task_id"].astype(str))
    if set(seen_effects) != expected_seen:
        raise IntegrityFailure("E177 seen-effect task set changed")
    forbidden = set(tasks.loc[tasks.target_split.isin(["calibration", "evaluation"]), "task_id"].astype(str))
    if set(seen_effects) & forbidden:
        raise IntegrityFailure("calibration/evaluation truth entered E177 F2")

    coexpression = pd.read_csv(root / "TRAIN_CONTROL_COEXPRESSION_EDGES.csv")
    if not {"source", "target", "importance"}.issubset(coexpression.columns) or coexpression.empty:
        raise IntegrityFailure("E177 coexpression edge schema failed")
    tokens = set(panel.scgpt_token.astype(str))
    if not set(coexpression.source.astype(str)).issubset(tokens) or not set(coexpression.target.astype(str)).issubset(tokens):
        raise IntegrityFailure("E177 coexpression edge outside panel")
    if not np.isfinite(coexpression.importance.to_numpy(float)).all():
        raise IntegrityFailure("E177 coexpression weights are non-finite")

    access = pd.read_csv(root / "ROW_ACCESS_AUDIT.csv", keep_default_na=False)
    if access.truth_access_phase.value_counts().to_dict() != {
        "PRETRUTH_TRAIN_X": 8146,
        "PRETRUTH_CONTROL_X": 2500,
        "PRETRUTH_VALIDATION_X": 1193,
    }:
        raise IntegrityFailure("E177 F2 access phase counts changed")
    attestation = json.loads((root / "ACCESS_ATTESTATION.json").read_text())
    required = {
        "status": "PASS",
        "stage": "F2_PRETRUTH_ISOLATED_ASSET_BUILD",
        "calibration_target_x_rows_read": 0,
        "evaluation_target_x_rows_read": 0,
        "n_seen_train_validation_effects": N_TRAIN_TASKS + N_VALIDATION_TASKS,
        "n_query_only_tasks_without_y": N_HIDDEN_TASKS,
    }
    changed = {key: (value, attestation.get(key)) for key, value in required.items() if attestation.get(key) != value}
    if changed:
        raise IntegrityFailure(f"E177 F2 attestation changed: {changed}")
    asset_head = str(attestation.get("current_git_head", ""))
    if subprocess.run(["git", "cat-file", "-e", f"{asset_head}^{{commit}}"], cwd=ROOT, check=False).returncode:
        raise IntegrityFailure("E177 F2 asset-build commit is unavailable")
    if subprocess.run(["git", "merge-base", "--is-ancestor", asset_head, git_text("rev-parse", "HEAD")], cwd=ROOT, check=False).returncode:
        raise IntegrityFailure("current HEAD does not descend from E177 F2 asset build commit")
    return Assets(panel, tasks, controls, seen_effects, coexpression, attestation, sha256_file(root / "MANIFEST.sha256"), hashes)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-12 else 0.0


def quantize(values: np.ndarray, tolerance: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise IntegrityFailure("cannot quantize non-finite values")
    return np.rint(values / tolerance).astype(np.int64)


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
    sums = ranks.sum(axis=0)
    numerator = 12.0 * float(np.sum((sums - m * (n + 1) / 2.0) ** 2))
    tie_correction = 0.0
    for row in values:
        counts = np.unique(row, return_counts=True)[1]
        tie_correction += float(np.sum(counts**3 - counts))
    denominator = m * m * (n**3 - n) - m * tie_correction
    return numerator / denominator if denominator > 0 else float("nan")


def five_seed_stability(risks: np.ndarray, seed: int, n_boot: int = G4_BOOTSTRAPS) -> dict[str, Any]:
    values = np.asarray(risks, float)
    if values.ndim != 2 or values.shape[0] < 3:
        raise IntegrityFailure("G4 requires at least three aligned risk vectors")
    pairs = [(i, j) for i in range(values.shape[0]) for j in range(i + 1, values.shape[0])]
    cor = np.asarray([spearman(values[i], values[j]) for i, j in pairs], float)
    rng = np.random.default_rng(seed)
    boot: list[float] = []
    for _ in range(n_boot):
        take = rng.integers(0, values.shape[1], values.shape[1])
        draw = [spearman(values[i, take], values[j, take]) for i, j in pairs]
        if np.isfinite(draw).all():
            boot.append(float(np.median(draw)))
    lower = float(np.quantile(boot, 0.025)) if boot else float("nan")
    return {
        "n_estimators": int(values.shape[0]),
        "n_pairwise_correlations": len(pairs),
        "minimum_pairwise_spearman": float(np.nanmin(cor)),
        "median_pairwise_spearman": float(np.nanmedian(cor)),
        "maximum_pairwise_spearman": float(np.nanmax(cor)),
        "kendall_w": kendall_w(values),
        "bootstrap_valid": len(boot),
        "bootstrap_ci95_lower": lower,
        "passed": bool(math.isfinite(float(np.nanmedian(cor))) and float(np.nanmedian(cor)) >= 0.5 and math.isfinite(lower) and lower > 0),
    }


def score_gate(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, float)
    finite = bool(np.isfinite(values).all())
    levels = np.unique(quantize(values, SCORE_TOL)) if finite else np.asarray([])
    std = float(np.std(values, ddof=0)) if finite else float("nan")
    return {"all_finite": finite, "n_levels": int(len(levels)), "std": std, "passed": bool(finite and len(levels) >= 2 and std > SCORE_TOL)}


def predictor_gate(matrix: np.ndarray) -> dict[str, Any]:
    values = np.asarray(matrix, float)
    finite = bool(values.ndim == 2 and np.isfinite(values).all())
    if not finite:
        return {"all_finite": False, "n_unique_vectors": 0, "max_coordinate_std": float("nan"), "passed": False}
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


def weak_order_identical(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(rankdata(quantize(a, SCORE_TOL), method="average"), rankdata(quantize(b, SCORE_TOL), method="average")))


def stable_seed(*parts: str) -> int:
    return int(hashlib.sha256("\0".join(parts).encode()).hexdigest()[:8], 16)


def zscore_by_ref(values: pd.Series, reference: pd.Series) -> pd.Series:
    code_root = ROOT / "code/20260426_154505_perturb_transport_final_push"
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    from safetrans_confidence.scoring.protocol_v0_2 import zscore_by_ref as zscore

    result = zscore(values, reference)
    if len(result) != len(values) or not np.isfinite(result.to_numpy(float)).all():
        raise IntegrityFailure("zscore_by_ref returned invalid values")
    return result


def synthetic_tests() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(test_id: str, passed: bool, observed: str) -> None:
        rows.append({"test_id": test_id, "passed": bool(passed), "observed": observed})

    x = np.linspace(0, 1, 100)
    add("S1_constant_score_fails", not score_gate(np.ones(100))["passed"], "constant")
    add("S2_variable_score_passes", score_gate(x)["passed"], "variable")
    add("S3_constant_predictor_fails", not predictor_gate(np.ones((100, 8)))["passed"], "constant predictor")
    add("S4_variable_predictor_passes", predictor_gate(np.column_stack([x, x**2]))["passed"], "variable predictor")
    stable = np.stack([x, x + 0.001 * np.sin(np.arange(100)), x + 0.001 * np.cos(np.arange(100)), x + 0.001, x + 0.002])
    add("S5_stable_five_seed_gate_passes", five_seed_stability(stable, 17, n_boot=250)["passed"], "stable")
    rng = np.random.default_rng(18)
    unstable = np.stack([rng.permutation(x) for _ in range(5)])
    add("S6_unstable_five_seed_gate_fails", not five_seed_stability(unstable, 19, n_boot=250)["passed"], "unstable")
    add("S7_magnitude_clone_detected", weak_order_identical(x, 2 * x + 1), "clone=True")
    return pd.DataFrame(rows)


def build_graphs(assets: Assets) -> tuple[dict[str, list[Any]], list[Any], pd.DataFrame]:
    import torch
    from torch_geometric.data import Data

    target_position = {
        str(row.gene_name): int(row.panel_index)
        for row in assets.panel.itertuples(index=False)
        if str(row.panel_role) == "REGISTERED_TARGET"
    }
    if len(target_position) != N_TARGETS:
        raise IntegrityFailure("registered target genes are not all in panel")
    supervised = {"train": [], "validation": []}
    query: list[Any] = []
    audit: list[dict[str, Any]] = []
    for row in assets.tasks.itertuples(index=False):
        task_id = str(row.task_id)
        group = str(row.technical_group)
        basal = assets.controls[f"G{group}"]
        flag = np.zeros(N_GENES, np.float32)
        flag[target_position[str(row.perturbation)]] = 1.0
        x = torch.from_numpy(np.stack([basal, flag], axis=1))
        query_graph = Data(
            x=x,
            pert=task_id,
            technical_group=group,
            perturbation=str(row.perturbation),
            target_split=str(row.target_split),
        )
        if getattr(query_graph, "y", None) is not None:
            raise IntegrityFailure("query graph unexpectedly contains y")
        query.append(query_graph)
        audit.append({"task_id": task_id, "graph_role": "query", "target_split": str(row.target_split), "contains_y": False})
        if task_id in assets.seen_effects:
            target = basal + assets.seen_effects[task_id]
            split = str(row.target_split)
            graph = Data(
                x=x,
                y=torch.from_numpy(target).unsqueeze(0),
                pert=task_id,
                technical_group=group,
                perturbation=str(row.perturbation),
                target_split=split,
            )
            if split not in supervised:
                raise IntegrityFailure(f"seen effect is not train/validation: {task_id}")
            supervised[split].append(graph)
            audit.append({"task_id": task_id, "graph_role": f"supervised_{split}", "target_split": split, "contains_y": True})
    if len(supervised["train"]) != N_TRAIN_TASKS or len(supervised["validation"]) != N_VALIDATION_TASKS:
        raise IntegrityFailure("E177 supervised graph counts changed")
    if len(query) != N_TARGETS * len(TECH_GROUPS):
        raise IntegrityFailure("E177 query graph count changed")
    hidden_ids = set(assets.tasks.loc[assets.tasks.target_split.isin(["calibration", "evaluation"]), "task_id"].astype(str))
    if any(str(graph.pert) in hidden_ids and getattr(graph, "y", None) is not None for graph in query):
        raise IntegrityFailure("hidden query graph contains y")
    return supervised, query, pd.DataFrame(audit)


def query_batch_size(batch: Any) -> int:
    if not hasattr(batch, "pert"):
        raise IntegrityFailure("query batch lacks task ids")
    if hasattr(batch, "y") and getattr(batch, "y") is not None:
        raise IntegrityFailure("query batch unexpectedly contains y")
    return len(batch.pert)


def scgpt_forward_no_y(model: Any, batch: Any, gene_ids: np.ndarray, device: Any, amp: bool, e65: Any) -> Any:
    import torch

    size = query_batch_size(batch)
    batch = batch.to(device)
    values = batch.x[:, 0].view(size, N_GENES)
    flags = batch.x[:, 1].long().view(size, N_GENES)
    raw_index = torch.arange(N_GENES, device=device, dtype=torch.long)
    mapped = e65.map_raw_id_to_vocab_id(raw_index, gene_ids).repeat(size, 1)
    mask = torch.zeros_like(values, dtype=torch.bool, device=device)
    with torch.cuda.amp.autocast(enabled=amp):
        return model(mapped, values, flags, src_key_padding_mask=mask, CLS=False, CCE=False, MVC=False, ECS=False, do_sample=False)["mlm_output"]


def train_scgpt(seed_value: int, supervised: dict[str, list[Any]], query: list[Any], genes: list[str], device: Any) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, Any]]:
    import torch
    from torch_geometric.loader import DataLoader

    set_seed(seed_value)
    e65 = import_script(f"e65_for_e177_{seed_value}", E65_SCRIPT)
    model, _, meta = e65.load_model(device)
    gene_ids = e65.make_gene_ids(genes, meta["vocab"])
    train_loader = DataLoader(supervised["train"], 16, shuffle=True, generator=torch.Generator().manual_seed(seed_value))
    val_loader = DataLoader(supervised["validation"], 16, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_loss, best_epoch, stale, best_state = float("inf"), 0, 0, None
    history: list[dict[str, Any]] = []
    for epoch in range(1, 11):
        train_loss = e65.train_one_epoch(model, train_loader, gene_ids, optimizer, scaler, device, device.type == "cuda")
        val_loss = e65.evaluate_mse(model, val_loader, gene_ids, device, device.type == "cuda")
        history.append({"seed": seed_value, "model": "scGPT", "epoch": epoch, "train_mse": train_loss, "validation_mse": val_loss})
        print(f"E177 scGPT seed={seed_value} epoch={epoch} train={train_loss:.6f} val={val_loss:.6f}", flush=True)
        if val_loss < best_loss - 1e-7:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 3:
                break
    if best_state is None:
        raise IntegrityFailure("scGPT produced no validation checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()
    predictions: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for batch in DataLoader(query, 16, shuffle=False):
            output = scgpt_forward_no_y(model, batch, gene_ids, device, device.type == "cuda", e65)
            moved = batch.to(device)
            basal = moved.x[:, 0].view(len(batch.pert), N_GENES)
            for task_id, pred, base in zip(batch.pert, output.detach().cpu().numpy(), basal.detach().cpu().numpy()):
                predictions[str(task_id)] = np.asarray(pred - base, np.float32)
    return predictions, pd.DataFrame(history), {
        "best_epoch": best_epoch,
        "best_validation_mse": best_loss,
        "matched_pretrained_parameter_tensors": meta["matched_pretrained_parameter_tensors"],
    }


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


def gears_mse_epoch(model: Any, loader: Any, device: Any, optimizer: Any | None) -> float:
    import torch

    model.train(optimizer is not None)
    losses = []
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for batch in loader:
            batch = batch.to(device)
            prediction = model(batch)
            loss = torch.mean((prediction - batch.y) ** 2)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def train_gears(seed_value: int, supervised: dict[str, list[Any]], query: list[Any], genes: list[str], coexpression: pd.DataFrame, device: Any) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, Any]]:
    import torch
    from torch_geometric.loader import DataLoader
    from gears.model import GEARS_Model

    set_seed(seed_value)
    go_index, go_weight, n_go = go_edges(genes)
    co_index, co_weight = edge_tensors(coexpression, genes)
    model = GEARS_Model({
        "hidden_size": 64,
        "num_go_gnn_layers": 1,
        "num_gene_gnn_layers": 1,
        "decoder_hidden_size": 16,
        "uncertainty": False,
        "G_go": go_index,
        "G_go_weight": go_weight,
        "G_coexpress": co_index,
        "G_coexpress_weight": co_weight,
        "device": str(device),
        "num_genes": N_GENES,
    }).to(device)
    train_loader = DataLoader(supervised["train"], 16, shuffle=True, generator=torch.Generator().manual_seed(seed_value))
    val_loader = DataLoader(supervised["validation"], 16, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_loss, best_epoch, stale, best_state = float("inf"), 0, 0, None
    history: list[dict[str, Any]] = []
    for epoch in range(1, 41):
        train_loss = gears_mse_epoch(model, train_loader, device, optimizer)
        val_loss = gears_mse_epoch(model, val_loader, device, None)
        history.append({"seed": seed_value, "model": "GEARS", "epoch": epoch, "train_mse": train_loss, "validation_mse": val_loss})
        print(f"E177 GEARS seed={seed_value} epoch={epoch} train={train_loss:.6f} val={val_loss:.6f}", flush=True)
        if val_loss < best_loss - 1e-7:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 6:
                break
    if best_state is None:
        raise IntegrityFailure("GEARS produced no validation checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()
    predictions: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for batch in DataLoader(query, 16, shuffle=False):
            if getattr(batch, "y", None) is not None:
                raise IntegrityFailure("GEARS query batch unexpectedly contains y")
            size = query_batch_size(batch)
            batch = batch.to(device)
            output = model(batch)
            basal = batch.x[:, 0].view(size, N_GENES)
            for task_id, pred, base in zip(batch.pert, output.detach().cpu().numpy(), basal.detach().cpu().numpy()):
                predictions[str(task_id)] = np.asarray(pred - base, np.float32)
    return predictions, pd.DataFrame(history), {
        "best_epoch": best_epoch,
        "best_validation_mse": best_loss,
        "n_go_edges": n_go,
        "n_coexpression_edges": len(coexpression),
    }


def runtime_environment(device: Any) -> dict[str, Any]:
    import scipy
    import torch
    import torch_geometric

    gpu = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(index)
            gpu.append({"index": index, "name": prop.name, "total_memory_bytes": int(prop.total_memory)})
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
        "selected_device": str(device),
        "gpu_inventory": gpu,
    }


def assemble_scores(assets: Assets, predictions: dict[str, dict[str, np.ndarray]]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    task_order = assets.tasks.task_id.astype(str).tolist()
    expected_predictors = {
        *(f"scGPT_seed{seed}" for seed in SEEDS),
        *(f"GEARS_seed{seed}" for seed in SEEDS),
    }
    if set(predictions) != expected_predictors:
        raise IntegrityFailure("prediction family/seed set changed")
    for name, mapping in predictions.items():
        if set(mapping) != set(task_order):
            raise IntegrityFailure(f"prediction coverage changed for {name}")
    arrays = {name: np.stack([mapping[task] for task in task_order]).astype(np.float32) for name, mapping in predictions.items()}
    arrays["scGPT_seed_mean"] = np.mean(np.stack([arrays[f"scGPT_seed{seed}"] for seed in SEEDS]), axis=0).astype(np.float32)
    arrays["GEARS_seed_mean"] = np.mean(np.stack([arrays[f"GEARS_seed{seed}"] for seed in SEEDS]), axis=0).astype(np.float32)
    arrays["ensemble_seed_family_mean"] = ((arrays["scGPT_seed_mean"] + arrays["GEARS_seed_mean"]) / 2).astype(np.float32)

    scores = assets.tasks.copy()
    controls = list(assets.controls.values())
    scores["technical_control_similarity_max"] = [
        max(cosine(assets.controls[f"G{row.technical_group}"], ref) for ref in controls)
        for row in scores.itertuples(index=False)
    ]
    trained_targets = set(scores.loc[scores.target_split.eq("train"), "perturbation"].astype(str))
    scores["perturbation_support_count"] = [
        int(sum(1 for group in TECH_GROUPS if str(row.perturbation) in trained_targets))
        if str(row.perturbation) in trained_targets else 0
        for row in scores.itertuples(index=False)
    ]
    scores["model_disagreement_rmse"] = np.sqrt(np.mean((arrays["scGPT_seed_mean"] - arrays["GEARS_seed_mean"]) ** 2, axis=1))
    scores["predicted_magnitude"] = np.sqrt(np.mean(arrays["ensemble_seed_family_mean"] ** 2, axis=1))
    train = scores.target_split.eq("train")
    if int(train.sum()) != N_TRAIN_TASKS:
        raise IntegrityFailure("E177 train reference count changed")
    scores["z_log_support_train"] = zscore_by_ref(np.log1p(scores.perturbation_support_count), np.log1p(scores.loc[train, "perturbation_support_count"]))
    scores["z_disagreement_train"] = zscore_by_ref(scores.model_disagreement_rmse, scores.loc[train, "model_disagreement_rmse"])
    scores["safeconf_confidence"] = scores.z_log_support_train - scores.z_disagreement_train
    scores["safeconf_risk"] = -scores.safeconf_confidence
    scores["true_error_rmse"] = np.nan
    scores["calibration_or_evaluation_truth_present"] = False

    seed_risks: dict[int, pd.Series] = {}
    for omitted in SEEDS:
        kept = [seed for seed in SEEDS if seed != omitted]
        sc = np.mean(np.stack([arrays[f"scGPT_seed{seed}"] for seed in kept]), axis=0)
        ge = np.mean(np.stack([arrays[f"GEARS_seed{seed}"] for seed in kept]), axis=0)
        disagreement = np.sqrt(np.mean((sc - ge) ** 2, axis=1))
        z_dis = zscore_by_ref(pd.Series(disagreement), pd.Series(disagreement[train.to_numpy()]))
        seed_risks[omitted] = -(scores.z_log_support_train - z_dis)
        scores[f"seed_risk_{omitted}"] = seed_risks[omitted]
    return scores, arrays


def run_gate(scores: pd.DataFrame, arrays: dict[str, np.ndarray], graph_audit: pd.DataFrame) -> dict[str, Any]:
    hidden = scores.target_split.isin(["calibration", "evaluation"])
    if int(hidden.sum()) != N_HIDDEN_TASKS:
        raise IntegrityFailure("E177 hidden query count changed")
    array_index = {task: i for i, task in enumerate(scores.task_id.astype(str))}
    blocks: dict[str, pd.Series] = {
        "hidden_all": hidden,
        "calibration": scores.target_split.eq("calibration"),
        "evaluation": scores.target_split.eq("evaluation"),
    }
    for group in TECH_GROUPS:
        blocks[f"hidden_G{group}"] = hidden & scores.technical_group.astype(str).eq(group)

    g2_rows: list[dict[str, Any]] = []
    g3_rows: list[dict[str, Any]] = []
    g4_rows: list[dict[str, Any]] = []
    g5_rows: list[dict[str, Any]] = []
    predictors = [*(f"scGPT_seed{seed}" for seed in SEEDS), *(f"GEARS_seed{seed}" for seed in SEEDS), "scGPT_seed_mean", "GEARS_seed_mean", "ensemble_seed_family_mean"]
    for name, mask in blocks.items():
        block = scores.loc[mask].copy()
        if block.empty:
            raise IntegrityFailure(f"empty E177 gate block: {name}")
        g2_rows.append({"block": name, "n_tasks": len(block), **score_gate(block.safeconf_risk.to_numpy(float))})
        risk_matrix = np.stack([block[f"seed_risk_{seed}"].to_numpy(float) for seed in SEEDS])
        g4_rows.append({"block": name, "n_tasks": len(block), **five_seed_stability(risk_matrix, stable_seed("E177_G4", name))})
        g5_rows.append({
            "block": name,
            "n_tasks": len(block),
            "risk_magnitude_operational_weak_order_identical": weak_order_identical(block.safeconf_risk.to_numpy(float), block.predicted_magnitude.to_numpy(float)),
        })
        take = np.asarray([array_index[task] for task in block.task_id.astype(str)], int)
        for predictor in predictors:
            g3_rows.append({"block": name, "predictor_name": predictor, "n_tasks": len(block), **predictor_gate(arrays[predictor][take])})
    synthetic = synthetic_tests()
    query_with_y = graph_audit.graph_role.eq("query") & strict_bool(graph_audit.contains_y, "contains_y")
    g1_passed = bool(
        assets_boundary := (
            int(query_with_y.sum()) == 0
            and not scores.loc[hidden, "calibration_or_evaluation_truth_present"].astype(bool).any()
        )
    )
    g2 = pd.DataFrame(g2_rows)
    g3 = pd.DataFrame(g3_rows)
    g4 = pd.DataFrame(g4_rows)
    g5 = pd.DataFrame(g5_rows)
    pretruth_pass = bool(
        g1_passed
        and g2.passed.astype(bool).all()
        and g3.passed.astype(bool).all()
        and g4.passed.astype(bool).all()
        and synthetic.passed.astype(bool).all()
    )
    return {"g1_passed": g1_passed, "g2": g2, "g3": g3, "g4": g4, "g5": g5, "synthetic": synthetic, "pretruth_pass": pretruth_pass}


def write_release(assets: Assets, scores: pd.DataFrame, arrays: dict[str, np.ndarray], graph_audit: pd.DataFrame, histories: pd.DataFrame, model_audit: pd.DataFrame, gate: dict[str, Any], head: str, branch: str, remote_heads: dict[str, str], frozen_hashes: list[dict[str, Any]], environment: dict[str, Any], wall_seconds: float) -> Path:
    if RELEASE.exists() or STAGING.exists():
        raise IntegrityFailure("E177 pretruth release is append-only and already exists")
    try:
        for sub in ("tables", "arrays", "reports"):
            (STAGING / sub).mkdir(parents=True, exist_ok=False)
        atomic_csv(STAGING / "tables/PRETRUTH_SCORING_INTERFACE.csv", scores)
        atomic_npz(STAGING / "arrays/PRETRUTH_PREDICTIONS.npz", arrays)
        atomic_csv(STAGING / "tables/QUERY_GRAPH_AUDIT.csv", graph_audit)
        atomic_csv(STAGING / "tables/TRAINING_HISTORY.csv", histories)
        atomic_csv(STAGING / "tables/MODEL_AUDIT.csv", model_audit)
        atomic_csv(STAGING / "tables/INPUT_HASHES.csv", pd.DataFrame(frozen_hashes + assets.input_hashes))
        atomic_csv(STAGING / "tables/G2_SCORE_CERTIFICATES.csv", gate["g2"])
        atomic_csv(STAGING / "tables/G3_PREDICTOR_CERTIFICATES.csv", gate["g3"])
        atomic_csv(STAGING / "tables/G4_SEED_STABILITY.csv", gate["g4"])
        atomic_csv(STAGING / "tables/G5_MAGNITUDE_EQUIVALENCE.csv", gate["g5"])
        atomic_csv(STAGING / "tables/SYNTHETIC_REGRESSION_TESTS.csv", gate["synthetic"])
        atomic_json(STAGING / "RUNTIME_ENVIRONMENT.json", environment)
        files = sorted(path for path in STAGING.rglob("*") if path.is_file())
        file_hashes = {path.relative_to(STAGING).as_posix(): sha256_file(path) for path in files}
        snapshot = {
            "schema": "safeconf_e177_pretruth_gate_snapshot_v1",
            "experiment": EXPERIMENT,
            "stage": "F3_PRETRUTH_GATE",
            "status": "PASS" if gate["pretruth_pass"] else "FAIL",
            "all_registered_gates_passed": bool(gate["pretruth_pass"]),
            "created_at": now(),
            "git_head": head,
            "git_branch": branch,
            "code_freeze_remote_heads": remote_heads,
            "runner_sha256": sha256_file(RUNNER),
            "asset_builder_sha256": sha256_file(ASSET_BUILDER),
            "f2_manifest_sha256": assets.manifest_sha256,
            "logical_x_rows_read_by_f2": assets.attestation["logical_x_rows_read_by_phase"],
            "calibration_target_x_rows_read": 0,
            "evaluation_target_x_rows_read": 0,
            "n_train_tasks": int(scores.target_split.eq("train").sum()),
            "n_validation_tasks": int(scores.target_split.eq("validation").sum()),
            "n_hidden_query_tasks": int(scores.target_split.isin(["calibration", "evaluation"]).sum()),
            "query_graphs_containing_y": int((graph_audit.graph_role.eq("query") & strict_bool(graph_audit.contains_y, "contains_y")).sum()),
            "registered_g2_units": int(len(gate["g2"])),
            "registered_g4_units": int(len(gate["g4"])),
            "synthetic_tests_passed": int(gate["synthetic"].passed.astype(bool).sum()),
            "pretruth_files_sha256": file_hashes,
            "wall_seconds": float(wall_seconds),
            "public_processed_data_only": True,
            "operational_wetlab_protocol_in_scope": False,
            "deployment_authorized": False,
        }
        atomic_json(STAGING / "PRETRUTH_GATE_SNAPSHOT.json", snapshot)
        report = (
            "# E177 pretruth gate report\n\n"
            f"Status: **{snapshot['status']}**.\n\n"
            f"Train tasks: {snapshot['n_train_tasks']}; validation tasks: {snapshot['n_validation_tasks']}; "
            f"hidden query tasks: {snapshot['n_hidden_query_tasks']}.\n\n"
            "Calibration and evaluation target vectors remain sealed. This gate uses only predictions, "
            "training/validation supervised effects, and query graphs without `y`.\n"
        )
        atomic_bytes(STAGING / "reports/E177_PRETRUTH_GATE_REPORT.md", report.encode())
        os.replace(STAGING, RELEASE)
        return RELEASE / "PRETRUTH_GATE_SNAPSHOT.json"
    except Exception:
        shutil.rmtree(STAGING, ignore_errors=True)
        raise


def run_formal(asset_root: Path, device_name: str) -> dict[str, Any]:
    started = time.time()
    head, branch, remote_heads, frozen_hashes = formal_input_audit()
    assets = load_assets(asset_root)
    supervised, query, graph_audit = build_graphs(assets)
    import torch

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise IntegrityFailure("registered CUDA device is unavailable")
    genes = assets.panel.scgpt_token.astype(str).tolist()
    environment = runtime_environment(device)
    predictions: dict[str, dict[str, np.ndarray]] = {}
    histories: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for seed_value in SEEDS:
        sc, hist, audit = train_scgpt(seed_value, supervised, query, genes, device)
        predictions[f"scGPT_seed{seed_value}"] = sc
        histories.append(hist)
        audits.append({"seed": seed_value, "model": "scGPT", **audit})
        if device.type == "cuda":
            torch.cuda.empty_cache()
        ge, hist, audit = train_gears(seed_value, supervised, query, genes, assets.coexpression, device)
        predictions[f"GEARS_seed{seed_value}"] = ge
        histories.append(hist)
        audits.append({"seed": seed_value, "model": "GEARS", **audit})
        if device.type == "cuda":
            torch.cuda.empty_cache()
    scores, arrays = assemble_scores(assets, predictions)
    gate = run_gate(scores, arrays, graph_audit)
    snapshot = write_release(
        assets,
        scores,
        arrays,
        graph_audit,
        pd.concat(histories, ignore_index=True),
        pd.DataFrame(audits),
        gate,
        head,
        branch,
        remote_heads,
        frozen_hashes,
        environment,
        time.time() - started,
    )
    return {
        "status": "PASS" if gate["pretruth_pass"] else "FAIL",
        "snapshot": str(snapshot.relative_to(ROOT)),
        "snapshot_sha256": sha256_file(snapshot),
        "wall_seconds": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=ASSET_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--synthetic-test-only", action="store_true")
    args = parser.parse_args()
    if args.synthetic_test_only:
        tests = synthetic_tests()
        print(tests.to_string(index=False))
        if len(tests) != 7 or not tests.passed.astype(bool).all():
            raise SystemExit(2)
        return
    result = run_formal(args.asset_root, args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
