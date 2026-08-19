#!/usr/bin/env python3
"""Complete the pre-frozen H1 component-gene LOGO audit from E165 tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
E165 = ROOT / "docs/实验结果/E165_wessels_truth_unseal_evaluation_20260715"
OUT = ROOT / "docs/实验结果/E165a_wessels_h1_logo_completion_20260716"
CONTRACT = OUT / "ANALYSIS_CONTRACT.md"
STAGING = OUT / ".release.staging"
RELEASE = OUT / "release"

STATUS = E165 / "release/RUN_STATUS.json"
MANIFEST = E165 / "release/RESULTS_SHA256.csv"
METRICS = E165 / "release/tables/E165_PREDICTOR_TASK_METRICS.csv.gz"
HYPOTHESES = E165 / "release/tables/E165_HYPOTHESIS_TESTS.csv"
TASKS = E165 / "release/tables/E165_TEST_TRUTH_TASKS.csv"

EXPECTED = {
    STATUS: "51ae8e62f9930c941138a2cfd6c099c24bb19b2a992aecdbf09a9f093c5646a7",
    MANIFEST: "e86b7a48aee7adbc508f4392592247fdad33cf30730f801c819e05c50fca443e",
    METRICS: "c37467db93d5f69ea33851ace13fc4c9c63747976bb8763f3e741813ea889ccd",
    HYPOTHESES: "df5e631a2c965aa535defef60eed25aa10bde6e1db58b7132a0d3c5b6ce9d0b4",
    TASKS: "a761c433814a5ffd5d464680c6f789743f0ecdccb289a2260c2d7189a618d0ba",
}
ALLOWLIST = {
    ".E165a_TRANSACTION.json",
    "RUN_STATUS.json",
    "RESULTS_SHA256.csv",
    "README_先看这个.md",
    "reports/E165a_REPORT.md",
    "tables/E165a_H1_LOGO.csv",
    "tables/E165a_INPUT_HASHES.csv",
}
EXPECTED_PYTHON = Path("/home/yyf/.conda/envs/prescribe_env/bin/python")


class IntegrityFailure(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("preflight", "formal"))
    return parser.parse_args()


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path, follow_symlinks=False)
    finally:
        temporary.unlink(missing_ok=True)
    fsync_dir(path.parent)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, json_bytes(value))


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_bytes(path, frame.to_csv(index=False, float_format="%.17g").encode("utf-8"))


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def git_gate(path: Path, head: str, *, require: bool) -> dict[str, Any]:
    relative = path.relative_to(ROOT).as_posix()
    try:
        committed = subprocess.check_output(
            ["git", "show", f"{head}:{relative}"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        if require:
            raise IntegrityFailure(f"Uncommitted required file: {relative}")
        return {"path": relative, "sha256": sha256_file(path), "matches_head": False}
    observed = sha256_file(path)
    matches = observed == hashlib.sha256(committed).hexdigest()
    if require and not matches:
        raise IntegrityFailure(f"Working file differs from HEAD: {relative}")
    return {"path": relative, "sha256": observed, "matches_head": matches}


def preflight(*, formal: bool) -> dict[str, Any]:
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve() or sys.version_info[:3] != (3, 9, 25):
        raise IntegrityFailure("E165a requires the frozen Python 3.9.25 environment")
    head = git_head()
    own = [git_gate(RUNNER, head, require=formal), git_gate(CONTRACT, head, require=formal)]
    rows = []
    for path, expected in EXPECTED.items():
        if not path.is_file() or path.is_symlink() or sha256_file(path) != expected:
            raise IntegrityFailure(f"Frozen E165 input changed: {path}")
        git_gate(path, head, require=True)
        rows.append({"path": path.relative_to(ROOT).as_posix(), "sha256": expected, "bytes": path.stat().st_size})
    status = json.loads(STATUS.read_text())
    if status.get("phase") != "complete_one_time_test_truth_evaluation":
        raise IntegrityFailure("E165 is not complete")
    return {"git_head": head, "own": own, "inputs": rows}


def compute_logo() -> tuple[pd.DataFrame, dict[str, Any]]:
    tasks = pd.read_csv(TASKS)
    metrics = pd.read_csv(METRICS)
    hypotheses = pd.read_csv(HYPOTHESES)
    order = tasks["condition"].astype(str).tolist()
    if len(order) != 48 or len(set(order)) != 48:
        raise IntegrityFailure("E165 task axis changed")
    blocks = {}
    for predictor in ("cell_weighted_perturbed_mean", "matching_single_mean"):
        block = metrics.loc[metrics["predictor"].eq(predictor)].copy()
        if block["condition"].astype(str).tolist() != order:
            raise IntegrityFailure(f"Metric order changed for {predictor}")
        blocks[predictor] = block["pca10_rmse"].to_numpy(float)
    delta = blocks["cell_weighted_perturbed_mean"] - blocks["matching_single_mean"]
    h1 = hypotheses.loc[hypotheses["hypothesis"].eq("H1")]
    if len(h1) != 1 or not np.isclose(delta.mean(), float(h1.iloc[0]["point_estimate"]), rtol=0, atol=1e-15):
        raise IntegrityFailure("Reconstructed H1 differs from E165")
    genes = sorted({gene for condition in order for gene in condition.split("+")})
    rows = []
    for gene in genes:
        keep = np.asarray([gene not in condition.split("+") for condition in order], dtype=bool)
        rows.append({
            "record_type": "per_gene",
            "removed_gene": gene,
            "removed_tasks": int((~keep).sum()),
            "remaining_tasks": int(keep.sum()),
            "mean_delta_rmse_cellweighted_minus_matching": float(delta[keep].mean()),
            "positive_favors_matching": True,
        })
    values = np.asarray([row["mean_delta_rmse_cellweighted_minus_matching"] for row in rows])
    rows.append({
        "record_type": "summary",
        "removed_gene": "__ALL_COMPONENT_GENES__",
        "removed_tasks": np.nan,
        "remaining_tasks": np.nan,
        "mean_delta_rmse_cellweighted_minus_matching": np.nan,
        "positive_favors_matching": True,
        "n_component_genes": len(genes),
        "logo_min": float(values.min()),
        "logo_median": float(np.median(values)),
        "logo_max": float(values.max()),
        "logo_fraction_positive": float(np.mean(values > 0)),
    })
    summary = {
        "full_H1_point": float(delta.mean()),
        "E165_H1_passed_unchanged": bool(h1.iloc[0]["passed"]),
        "n_component_genes": len(genes),
        "logo_min": float(values.min()),
        "logo_median": float(np.median(values)),
        "logo_max": float(values.max()),
        "logo_fraction_positive": float(np.mean(values > 0)),
    }
    return pd.DataFrame(rows), summary


def formal(pre: dict[str, Any]) -> dict[str, Any]:
    if RELEASE.exists() or STAGING.exists():
        raise IntegrityFailure("E165a is append-only")
    table, summary = compute_logo()
    STAGING.mkdir(parents=True)
    (STAGING / "reports").mkdir(); (STAGING / "tables").mkdir()
    transaction = {"schema": "safeconf_e165a_transaction_v1", "transaction_id": uuid.uuid4().hex, "created_at": now()}
    atomic_json(STAGING / ".E165a_TRANSACTION.json", transaction)
    atomic_csv(STAGING / "tables/E165a_H1_LOGO.csv", table)
    atomic_csv(STAGING / "tables/E165a_INPUT_HASHES.csv", pd.DataFrame(pre["inputs"]))
    report = (
        "# E165a H1 LOGO补充\n\n"
        f"H1完整均值为`{summary['full_H1_point']:.17g}`，E165通过状态保持`{summary['E165_H1_passed_unchanged']}`。\n\n"
        f"逐component gene删除后：min=`{summary['logo_min']:.17g}`，median=`{summary['logo_median']:.17g}`，"
        f"max=`{summary['logo_max']:.17g}`，正值比例=`{summary['logo_fraction_positive']:.17g}`。\n\n"
        "本补充只读取E165 task-level metrics；raw H5AD和单细胞expression均未打开。\n"
    )
    atomic_bytes(STAGING / "reports/E165a_REPORT.md", report.encode("utf-8"))
    atomic_bytes(STAGING / "README_先看这个.md", b"# E165a\n\nSee `reports/E165a_REPORT.md`.\n")
    manifest_rows = []
    for path in sorted(STAGING.rglob("*")):
        if path.is_symlink():
            raise IntegrityFailure("Symlink in E165a staging")
        if path.is_file() and path.name not in {"RUN_STATUS.json", "RESULTS_SHA256.csv"}:
            manifest_rows.append({"relative_path": path.relative_to(STAGING).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    atomic_csv(STAGING / "RESULTS_SHA256.csv", pd.DataFrame(manifest_rows))
    status = {
        "schema": "safeconf_e165a_h1_logo_completion_v1",
        "phase": "complete_postpublication_contract_completion_no_raw_access",
        "completed_at": now(),
        "git_head": pre["git_head"],
        "transaction_id": transaction["transaction_id"],
        "raw_Wessels_opened": False,
        "expression_rows_indexed": 0,
        "test_truth_profiles_reopened": False,
        **summary,
        "results_manifest_sha256": sha256_file(STAGING / "RESULTS_SHA256.csv"),
        "artifact_sha256": {row["relative_path"]: row["sha256"] for row in manifest_rows},
    }
    atomic_json(STAGING / "RUN_STATUS.json", status)
    observed = {path.relative_to(STAGING).as_posix() for path in STAGING.rglob("*") if path.is_file()}
    if observed != ALLOWLIST:
        raise IntegrityFailure(f"E165a allowlist mismatch: {sorted(observed ^ ALLOWLIST)}")
    for path in STAGING.rglob("*"):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    for directory in sorted([path for path in STAGING.rglob("*") if path.is_dir()], key=lambda value: len(value.parts), reverse=True):
        fsync_dir(directory)
    fsync_dir(STAGING); STAGING.replace(RELEASE); fsync_dir(OUT)
    return status


def main() -> None:
    args = parse_args()
    pre = preflight(formal=args.mode == "formal")
    result = (
        {"schema": "safeconf_e165a_preflight_v1", "phase": "committed_inputs_pass_no_raw_access", "git_head": pre["git_head"], "own": pre["own"]}
        if args.mode == "preflight" else formal(pre)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
