#!/usr/bin/env python3
"""Attach the registered SAMS post-processing to the v0.4.2 evidence package.

This script never fits a model and never reads a new truth table.  The SAMS
training process owns generation and scoring.  Once that process is terminal,
this script copies its sealed receipts into the existing v0.4.2 package and
updates only the component-decision metadata.  Cross-family results remain a
separate scope from the 212-task current-holdout system comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = ROOT / (
    "docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/"
    "research_closure_20261001/data_model_feedback_20261003_v1"
)
DEFAULT_OUT = DEFAULT_BASE / "system_evidence_v042"
SAMS_DIR = DEFAULT_BASE / "sams"
PERTURBMAP_DIR = DEFAULT_BASE / "perturbmap_experiment_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def copy_if_present(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    shutil.copy2(src, dst)
    return True


def terminal_sams_status(sams: Path) -> tuple[dict, str]:
    status_path = sams / "TRAINING_STATUS.json"
    status = read_json(status_path) if status_path.exists() else {"status": "MISSING"}
    completion = sams / "TRAINING_COMPLETION.json"
    if completion.exists():
        status.update({"completion": read_json(completion)})
    cross = sams / "CROSSFAMILY_STATUS.json"
    if cross.exists():
        status.update({"crossfamily": read_json(cross)})
    validation = sams / "VALIDATION_COMPETENCE.json"
    if validation.exists():
        status.update({"validation_competence": read_json(validation)})
    original = sams / "VALIDATION_ORIGINAL_512_COMPETENCE.json"
    if original.exists():
        status.update({"original_512_competence": read_json(original)})
    if completion.exists() or cross.exists() or validation.exists():
        return status, str(status.get("crossfamily", {}).get("status", "POSTPROCESSING_COMPLETE"))
    return status, str(status.get("status", "UNKNOWN"))


def update_claims(out: Path, sams_status: dict, cross_status: str) -> None:
    path = out / "CLAIM_EVIDENCE_MATRIX.csv"
    if not path.exists():
        return
    claims = pd.read_csv(path)
    mask = claims["claim"].astype(str).eq("SAMS cross-family evidence is complete")
    evidence = str(SAMS_DIR)
    if mask.any():
        if cross_status == "COMPLETE":
            claims.loc[mask, "status"] = "SEEN_CROSS_FAMILY_TRANSFER_COMPLETE"
            claims.loc[mask, "decision"] = "retain as cross-family stress evidence; not independent confirmation"
        elif cross_status == "UPSTREAM_NOT_YET_QUALIFIED":
            claims.loc[mask, "status"] = "UPSTREAM_COMPETENCE_GATE_FAIL"
            claims.loc[mask, "decision"] = "retain generation and gate audit; do not claim qualified transfer"
        else:
            claims.loc[mask, "status"] = f"SAMS_POSTPROCESS_{cross_status}"
            claims.loc[mask, "decision"] = "retain status and continue only registered diagnostics"
        claims.loc[mask, "evidence"] = evidence
    # Keep the background-transfer candidate separate from the current
    # same-background system claim.  It is a development/SEEN result and
    # must not silently become a promoted PublicRule claim.
    if PERTURBMAP_DIR.exists():
        claim = "PerturbMap background transfer improves SafeConf risk ranking"
        if not claims["claim"].astype(str).eq(claim).any():
            claims = pd.concat([claims, pd.DataFrame([{
                "claim": claim,
                "status": "CONDITIONAL_BACKGROUND_TRANSFER_DEVELOPMENT",
                "evidence": str(PERTURBMAP_DIR),
                "decision": "retain candidate; do not replace same-background PublicRule",
            }])], ignore_index=True)
    claims.to_csv(path, index=False, lineterminator="\n")


def update_decision(out: Path, sams_status: dict, cross_status: str) -> None:
    path = out / "COMPONENT_DECISION.json"
    decision = read_json(path) if path.exists() else {"contract": "SafeConf v0.4.2"}
    decision["status"] = "CURRENT_CONTRACT_AUDIT_COMPLETE_SAMS_POSTPROCESSED"
    decision.setdefault("decisions", {})
    if cross_status == "COMPLETE":
        decision["decisions"]["sams"] = "RETAIN_SEEN_CROSS_FAMILY_STRESS_EVIDENCE"
        decision["sams_conclusion"] = (
            "SAMS passed the registered generation/competence path and produced "
            "bidirectional retrospective transfer evidence; it is not an independent "
            "confirmation because the upstream and evaluation assets are SEEN."
        )
    elif cross_status == "UPSTREAM_NOT_YET_QUALIFIED":
        decision["decisions"]["sams"] = "RETAIN_GATE_FAILURE_AS_BOUNDARY"
        decision["sams_conclusion"] = (
            "SAMS generation was completed but did not pass the registered upstream "
            "competence gate; no cross-family transfer claim is promoted."
        )
    else:
        decision["decisions"]["sams"] = f"RETAIN_STATUS_{cross_status}"
        decision["sams_conclusion"] = "SAMS post-processing status is recorded without promotion."
    decision["sams_status"] = sams_status.get("status")
    decision["sams_postprocess"] = sams_status
    if PERTURBMAP_DIR.exists():
        perturb_decision = PERTURBMAP_DIR / "COMPONENT_DECISION.json"
        decision["perturbmap"] = {
            "status": "CONDITIONAL_BACKGROUND_TRANSFER_DEVELOPMENT",
            "artifact": str(PERTURBMAP_DIR),
            "component_decision": read_json(perturb_decision) if perturb_decision.exists() else None,
            "default_replacement": False,
        }
    write_json(path, decision)


def attach_receipts(out: Path, sams: Path) -> list[str]:
    copied: list[str] = []
    names = [
        "TRAINING_COMPLETION.json",
        "VALIDATION_PREDICTION_SEAL.json",
        "VALIDATION_COMPETENCE.json",
        "VALIDATION_ORIGINAL_512_COMPETENCE.json",
        "CROSSFAMILY_STATUS.json",
        "CROSSFAMILY_MACRO.csv",
        "CROSSFAMILY_PAIRED_DIFFERENCES.csv",
        "CROSSFAMILY_STRATA.csv",
        "CROSSFAMILY_CDF_AUDIT.csv",
        "CROSSFAMILY_SHUFFLE_AUDIT.csv",
        "CROSSFAMILY_FIT_AUDIT.csv",
        "HELDOUT_PREDICTION_SEAL.json",
    ]
    for name in names:
        if copy_if_present(sams / name, out / f"SAMS_{name}"):
            copied.append(name)
    if PERTURBMAP_DIR.exists():
        for name in [
            "COMPONENT_DECISION.json", "DECISION.md", "RECONSTRUCTION_METRICS.csv",
            "RISK_METRICS.csv", "PAIRED_BOOTSTRAP.csv", "ROUTE_AUDIT.csv",
            "INFORMATION_BUDGET.csv",
        ]:
            if copy_if_present(PERTURBMAP_DIR / name, out / f"PERTURBMAP_{name}"):
                copied.append(f"PERTURBMAP_{name}")
    return copied


def refresh_manifest(out: Path) -> None:
    path = out / "MANIFEST.json"
    manifest = read_json(path) if path.exists() else {"contract": "SafeConf v0.4.2"}
    manifest["post_sams_refresh_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    manifest["files"] = {
        p.name: sha256(p)
        for p in sorted(out.iterdir())
        if p.is_file() and p.name != "MANIFEST.json"
    }
    write_json(path, manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sams", type=Path, default=SAMS_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not args.sams.exists() or not args.out.exists():
        raise SystemExit("SAMS or v0.4.2 evidence directory is missing")
    status, cross_status = terminal_sams_status(args.sams)
    if not (args.sams / "TRAINING_COMPLETION.json").exists():
        raise SystemExit("SAMS training is not terminal; do not refresh evidence yet")
    copied = attach_receipts(args.out, args.sams)
    update_claims(args.out, status, cross_status)
    update_decision(args.out, status, cross_status)
    run_status_path = args.out / "RUN_STATUS.json"
    run_status = read_json(run_status_path) if run_status_path.exists() else {}
    run_status["status"] = "COMPLETE_WITH_SAMS_POSTPROCESS"
    run_status["sams_status"] = status.get("status")
    run_status["sams_crossfamily_status"] = cross_status
    run_status["sams_receipts"] = copied
    run_status["sams_artifact"] = str(args.sams)
    write_json(run_status_path, run_status)
    refresh_manifest(args.out)
    print(json.dumps({"status": "REFRESHED", "crossfamily_status": cross_status, "copied": copied}, ensure_ascii=False))


if __name__ == "__main__":
    main()
