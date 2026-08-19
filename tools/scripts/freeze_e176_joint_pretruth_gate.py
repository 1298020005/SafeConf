#!/usr/bin/env python3
"""Aggregate committed E176 panel gates without opening calibration or evaluation truth."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any
import uuid

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "docs/实验结果/E176_four_donor_fresh_confirmation_20260719"
OUT = EXPERIMENT / "pretruth_joint"
PANELS = ("H01", "H02", "H03", "H04")
SCRIPT = Path(__file__).resolve()
PRETRUTH_RUNNER = ROOT / "tools/scripts/run_e176_four_donor_panel_pretruth.py"
ASSET_BUILDER = ROOT / "tools/scripts/build_e176_four_donor_panel_assets.py"
FREEZER = ROOT / "tools/scripts/freeze_e176_four_donor_fresh_confirmation.py"


class IntegrityFailure(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )


def git_text(*args: str) -> str:
    return git(*args).stdout.decode().strip()


def require_committed(path: Path, head: str) -> dict[str, Any]:
    relative = path.relative_to(ROOT).as_posix()
    try:
        committed = git("show", f"{head}:{relative}").stdout
    except subprocess.CalledProcessError as exc:
        raise IntegrityFailure(f"uncommitted gate input: {relative}") from exc
    if committed != path.read_bytes():
        raise IntegrityFailure(f"local gate input changed: {relative}")
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode())


def main() -> None:
    if OUT.exists():
        raise IntegrityFailure(f"append-only joint gate exists: {OUT}")
    head = git_text("rev-parse", "HEAD")
    branch = git_text("rev-parse", "--abbrev-ref", "HEAD")
    remote_heads: dict[str, str] = {}
    for remote in ("origin", "github"):
        fetched = f"refs/remotes/{remote}/{branch}"
        result = git("fetch", "--quiet", remote, f"refs/heads/{branch}:{fetched}", check=False)
        if result.returncode:
            raise IntegrityFailure(f"cannot verify {remote}")
        remote_head = git_text("rev-parse", fetched)
        if git("merge-base", "--is-ancestor", head, remote_head, check=False).returncode:
            raise IntegrityFailure(f"current HEAD absent from {remote}/{branch}")
        remote_heads[remote] = remote_head

    inputs = [
        require_committed(path, head)
        for path in (
            SCRIPT,
            PRETRUTH_RUNNER,
            ASSET_BUILDER,
            FREEZER,
            EXPERIMENT / "RUN_STATUS.json",
            EXPERIMENT / "MODEL_INPUT_LOCK.json",
            EXPERIMENT / "STATISTICAL_ANALYSIS_LOCK.json",
            EXPERIMENT / "PREREG_ANALYSIS_PLAN.md",
        )
    ]
    rows: list[pd.DataFrame] = []
    panel_evidence: dict[str, Any] = {}
    source_sha: str | None = None
    for panel in PANELS:
        release = EXPERIMENT / "pretruth_release" / panel
        snapshot_path = release / "PRETRUTH_GATE_SNAPSHOT.json"
        inputs.append(require_committed(snapshot_path, head))
        snapshot = json.loads(snapshot_path.read_text())
        required = {
            "schema": f"safeconf_e176_{panel.lower()}_pretruth_gate_snapshot_v1",
            "experiment": f"E176_four_donor_fresh_confirmation::{panel}",
            "stage": f"E176_{panel}_F2_PRETRUTH_GATE",
            "status": "PASS",
            "all_registered_gates_passed": True,
            "test_targeting_x_values_read": 0,
            "forbidden_column_unseen_x_values_read": 0,
            "test_query_graphs_containing_y": 0,
            "g4_risk_estimator": "leave_one_seed_out_family_mean",
        }
        changed = {
            key: {"expected": value, "observed": snapshot.get(key)}
            for key, value in required.items() if snapshot.get(key) != value
        }
        if changed:
            raise IntegrityFailure(f"{panel} pretruth snapshot changed: {changed}")
        if source_sha is None:
            source_sha = str(snapshot["source_full_sha256"])
        elif snapshot["source_full_sha256"] != source_sha:
            raise IntegrityFailure("panel source SHA mismatch")
        for relative, expected in snapshot["pretruth_files_sha256"].items():
            path = release / relative
            inputs.append(require_committed(path, head))
            if sha256_file(path) != expected:
                raise IntegrityFailure(f"{panel} pretruth hash changed: {relative}")
        f2_manifest = (
            Path("/home/yyf/data/safeconf_external/primary_cd4_perturbseq_2025")
            / "isolated/E176" / panel / "F2_pretruth/MANIFEST.sha256"
        )
        if sha256_file(f2_manifest) != snapshot["f2_manifest_sha256"]:
            raise IntegrityFailure(f"{panel} F2 manifest changed")
        g4 = pd.read_csv(release / "tables/G4_SEED_STABILITY.csv")
        if (
            len(g4) != 6
            or not g4.passed.astype(bool).all()
            or not g4.n_leave_one_seed_out_estimators.astype(int).eq(5).all()
            or not g4.n_pairwise_correlations.astype(int).eq(10).all()
        ):
            raise IntegrityFailure(f"{panel} five-seed G4 gate is not exact PASS")
        g4.insert(0, "panel_id", panel)
        rows.append(g4)
        panel_evidence[panel] = {
            "snapshot_sha256": sha256_file(snapshot_path),
            "f2_manifest_sha256": snapshot["f2_manifest_sha256"],
            "minimum_median_pairwise_spearman": float(g4.median_pairwise_spearman.min()),
            "minimum_ci95_lower": float(g4.bootstrap_ci95_lower.min()),
            "g4_units_passed": int(g4.passed.astype(bool).sum()),
            "test_donor_targeting_x_values_read": 0,
        }

    combined = pd.concat(rows, ignore_index=True)
    if len(combined) != 24 or not combined.passed.astype(bool).all():
        raise IntegrityFailure("joint G4 gate did not pass 24/24")
    OUT.mkdir(parents=True)
    (OUT / "tables").mkdir()
    (OUT / "reports").mkdir()
    combined.to_csv(OUT / "tables/E176_FIVE_SEED_G4_ALL_PANELS.csv", index=False)
    pd.DataFrame(inputs).drop_duplicates("path").to_csv(
        OUT / "tables/INPUT_HASHES.csv", index=False
    )
    report = f"""# E176 joint pretruth gate

四个供体轮换面板全部通过。24/24 个 panel×state×stratum 五种子稳定性单元通过；最小中位 pairwise Spearman 为 {combined.median_pairwise_spearman.min():.3f}，最小 bootstrap 95% CI 下界为 {combined.bootstrap_ci95_lower.min():.3f}。

所有 test query graph 均不含 y，test donor targeting X、校准真值与最终评价真值的读取数均为 0。该门控只授权下一阶段构建 160 个预先分配的校准靶点，不授权读取 640 个最终评价靶点。
"""
    atomic_bytes(OUT / "reports/E176_JOINT_PRETRUTH_REPORT.md", report.encode())
    snapshot = {
        "schema": "safeconf_e176_joint_pretruth_gate_v1",
        "experiment": "E176_four_donor_fresh_confirmation",
        "stage": "F2_JOINT_PRETRUTH_GATE",
        "status": "PASS",
        "decision": "CALIBRATION_TRUTH_ACCESS_AUTHORIZED",
        "git_head": head,
        "git_branch": branch,
        "remote_heads": remote_heads,
        "panels": list(PANELS),
        "g4_units": 24,
        "g4_units_passed": 24,
        "minimum_median_pairwise_spearman": float(combined.median_pairwise_spearman.min()),
        "minimum_ci95_lower": float(combined.bootstrap_ci95_lower.min()),
        "source_full_sha256": source_sha,
        "panel_evidence": panel_evidence,
        "test_donor_targeting_x_values_read": 0,
        "calibration_targeting_x_values_read": 0,
        "evaluation_targeting_x_values_read": 0,
        "calibration_runner_authorized": True,
        "final_evaluator_authorized": False,
        "optional_panel_dropping_used": False,
        "g4_threshold_relaxed_after_result": False,
        "deployment_authorized": False,
    }
    atomic_json(OUT / "PRETRUTH_GATE_SNAPSHOT.json", snapshot)
    artifacts = sorted(path for path in OUT.rglob("*")
                       if path.is_file() and path.name != "MANIFEST.sha256")
    manifest = "".join(
        f"{sha256_file(path)}  {path.relative_to(OUT).as_posix()}\n" for path in artifacts
    )
    atomic_bytes(OUT / "MANIFEST.sha256", manifest.encode())
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
