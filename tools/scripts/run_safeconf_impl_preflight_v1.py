#!/usr/bin/env python3
"""Read-only preflight for the SafeConf implementation run.

This command binds the current inputs and resources before any new fit.  It is
deliberately conservative: it records dirty worktree state and protected
processes, but never resets, stages, overwrites, opens permanent test truth, or
fits a model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


RUN_ID = "safeconf_impl_20261004_v1"
ROOT = Path("/home/yyf/runtime_worktrees/e220_reviewer_closure_20260921")
RUNTIME = Path("/home/yyf/runtime_artifacts")
IMPL_RUNTIME = RUNTIME / RUN_ID
IMPLEMENTATION_DOC = ROOT / (
    "docs/实验结果/Stage2_mature_upstream_20260928/"
    "dual_memory_continual/research_closure_20261001/"
    "data_model_feedback_20261003_v1/implementation_v1"
)
FEEDBACK = RUNTIME / "safeconf_research_20261003/feedback_v1"
COMMON = RUNTIME / "safeconf_research_20261001/common_gene_axis"
RISK_CACHE = COMMON / "risk_cache"
SYSTEM = ROOT / (
    "docs/实验结果/Stage2_mature_upstream_20260928/"
    "dual_memory_continual/research_closure_20261001/"
    "data_model_feedback_20261003_v1/system_evidence_v042"
)
E258_ROOT = Path("/home/yyf/data/feng2025_candidate")
E258_DOC = ROOT / "docs/实验结果/E258_feng2025_independent_confirmation_20260924"
TABPFN_ENV = Path("/home/yyf/.venvs/safeconf-tabpfn-20261001/bin/python")
TABPFN_CKPT = RUNTIME / "safeconf_research_20261001/tabpfn_v2_probe/tabpfn-v2-regressor.ckpt"
TABPFN_CKPT_SHA = "2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736"
PERTEMA_ROOT = RUNTIME / "official_pertema_43c09a"
PERTEMA_COMMIT = "43c09a32e23d0ee2ae5dfbab21b2deeab27f1803"
PROTECTED_PIDS = [1873824, 2428149]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(cmd: list[str], cwd: Path | None = None) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=30, check=False)
        return p.returncode, p.stdout.strip()
    except Exception as exc:  # pragma: no cover - host-specific diagnostics
        return 127, f"{type(exc).__name__}: {exc}"


def json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, path)


def csv_write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def file_binding(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path), "exists": False}
    return {"path": str(path), "exists": True, "bytes": path.stat().st_size,
            "sha256": sha256(path)}


def parquet_contract(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path), "exists": False}
    frame = pd.read_parquet(path)
    result = {
        "path": str(path), "exists": True, "rows": len(frame),
        "task_id_unique": int(frame.task_id.nunique()) if "task_id" in frame else None,
        "gene_unique": int(frame.gene.nunique()) if "gene" in frame else None,
        "columns": list(frame.columns),
    }
    if "true_error_rmse" in frame:
        result["truth_missing"] = int(frame.true_error_rmse.isna().sum())
    return result


def process_snapshot() -> dict:
    out = {}
    rc, text = run(["ps", "-p", *map(str, PROTECTED_PIDS),
                    "-o", "pid,ppid,stat,etime,pcpu,pmem,comm"])
    out["protected_processes"] = {"returncode": rc, "text": text}
    rc, text = run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader"])
    out["gpu"] = {"returncode": rc, "text": text}
    return out


def source_cache_contract() -> dict:
    files = sorted(RISK_CACHE.glob("nested_*_Manual.parquet"))
    rows = []
    for path in files:
        frame = pd.read_parquet(path, columns=["task_id", "gene", "fold"])
        rows.append({"path": str(path), "rows": len(frame),
                     "tasks": int(frame.task_id.nunique()),
                     "genes": int(frame.gene.nunique()),
                     "folds": sorted(map(int, frame.fold.unique()))})
    return {"files": rows, "file_count": len(rows),
            "expected_manual_nested_files": 10,
            "expected_rows": 1808, "expected_genes": 575}


def independent_asset_rows() -> list[dict]:
    candidates = [
        ("E258_test_donor", E258_ROOT, "sealed; upstream ability gate currently not passed"),
        ("Replogle_K562_GWPS", Path("/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/ReplogleWeissman2022_K562_gwps.h5ad"), "requires frozen predictor audit"),
        ("Nadig_E239_seen", ROOT / "docs/实验结果/E239_nadig_third_gene_confirmation_20260923", "SEEN same-study robustness only"),
    ]
    rows = []
    for name, path, role in candidates:
        rows.append({"asset": name, "path": str(path), "exists": path.exists(),
                     "role": role, "eligible_for_independent_score": False,
                     "reason": "qualification is metadata-only in preflight"})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=IMPL_RUNTIME)
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    IMPLEMENTATION_DOC.mkdir(parents=True, exist_ok=True)

    rc, git_status = run(["git", "status", "--short", "--branch"], ROOT)
    _, git_head = run(["git", "rev-parse", "HEAD"], ROOT)
    _, official_head = run(["git", "-C", str(PERTEMA_ROOT), "rev-parse", "HEAD"])
    official_source = PERTEMA_ROOT / "src/pertema/run_estimator.py"

    binding = {
        "run_id": RUN_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "worktree": str(ROOT), "git_head": git_head,
        "git_status_returncode": rc, "git_status": git_status,
        "protected_processes": process_snapshot(),
        "inputs": {
            "holdout": parquet_contract(FEEDBACK / "HOLDOUT_FEATURES.parquet"),
            "feedback_pool": parquet_contract(FEEDBACK / "FIXED_POOL_FEATURES.parquet"),
            "native_control_features": parquet_contract(RUNTIME / "safeconf_research_20261001/native_control_reference/TASK_FEATURES.parquet"),
            "source_cache": source_cache_contract(),
            "system_manifest": file_binding(SYSTEM / "MANIFEST.json"),
            "component_decision": file_binding(SYSTEM / "COMPONENT_DECISION.json"),
            "resource_ledger": file_binding(SYSTEM.parent / "RESOURCE_LEDGER.json"),
            "e258_contract": file_binding(E258_DOC / "PREFLIGHT_AND_ANALYSIS_CONTRACT.md"),
            "e258_report": file_binding(E258_DOC / "DEV_REPORT_20260925_历史何时有用.md"),
        },
        "tabpfn": {
            "python": str(TABPFN_ENV), "checkpoint": file_binding(TABPFN_CKPT),
            "expected_checkpoint_sha256": TABPFN_CKPT_SHA,
            "version_probe": run([str(TABPFN_ENV), "-c", "import tabpfn; print(tabpfn.__version__)"])[1],
        },
        "pertema": {
            "root": str(PERTEMA_ROOT), "head": official_head,
            "expected_commit": PERTEMA_COMMIT,
            "source": file_binding(official_source),
        },
        "independent_assets": independent_asset_rows(),
        "permanent_test_truth_opened": False,
        "new_download_bytes": 0,
        "new_large_upstream_training": False,
        "status": "PASS",
    }

    # A mismatch is recorded as a failure receipt, never silently repaired.
    checks = []
    hold = binding["inputs"]["holdout"]
    pool = binding["inputs"]["feedback_pool"]
    cache = binding["inputs"]["source_cache"]
    checks.append({"check": "holdout_shape", "pass": hold.get("rows") == 212 and hold.get("gene_unique") == 152})
    checks.append({"check": "feedback_pool_shape", "pass": pool.get("rows") == 331 and pool.get("gene_unique") == 228})
    checks.append({"check": "source_nested_cache", "pass": cache.get("file_count") == 10})
    checks.append({"check": "tabpfn_checkpoint", "pass": binding["tabpfn"]["checkpoint"].get("sha256") == TABPFN_CKPT_SHA})
    checks.append({"check": "pertema_commit", "pass": official_head == PERTEMA_COMMIT})
    binding["checks"] = checks
    binding["status"] = "PASS" if all(c["pass"] for c in checks) else "FAIL"

    json_write(out / "PREFLIGHT_MANIFEST.json", binding)
    json_write(IMPLEMENTATION_DOC / "IMPLEMENTATION_CONFIG.json", binding)
    csv_write(out / "ASSET_ROLE_LEDGER.csv", [
        {"asset": "PublicRule", "role": "reuse", "status": "frozen_default"},
        {"asset": "Source_nested_cache", "role": "reuse", "status": "1808_tasks_575_genes"},
        {"asset": "Target_feedback_pool", "role": "reuse", "status": "331_rows_228_genes"},
        {"asset": "Current_holdout", "role": "evaluation_only", "status": "212_rows_152_genes"},
        {"asset": "Source_TabPFN", "role": "reuse", "status": "DEV_COMPLETE"},
        {"asset": "Legacy_PertEMA", "role": "excluded_audit", "status": "truth_contract_mismatch"},
        {"asset": "E258_test", "role": "sealed", "status": "upstream_gate_not_passed"},
    ])
    json_write(out / "RESOURCE_SNAPSHOT.json", {
        "status": binding["status"], "historical_resource_ledger": str(SYSTEM.parent / "RESOURCE_LEDGER.json"),
        "new_download_bytes": 0, "new_large_upstream_training": 0,
        "protected_pids": PROTECTED_PIDS, "gpu_snapshot": binding["protected_processes"]["gpu"],
    })
    csv_write(out / "TRUTH_CONTRACT_AUDIT.csv", [
        {"asset": "current_holdout", "rows": hold.get("rows"), "genes": hold.get("gene_unique"), "truth_missing": hold.get("truth_missing")},
        {"asset": "feedback_pool", "rows": pool.get("rows"), "genes": pool.get("gene_unique"), "truth_missing": pool.get("truth_missing")},
    ])
    csv_write(out / "INDEPENDENT_ASSET_AUDIT.csv", binding["independent_assets"])
    if binding["status"] != "PASS":
        json_write(out / "FAILURE_RECEIPT.json", {"status": "PREFLIGHT_FAILED", "checks": checks,
                                                     "action": "do not fit; repair only the identified input contract"})
        return 2
    print(json.dumps({"status": binding["status"], "output": str(out), "git_head": git_head}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
