#!/usr/bin/env python3
"""Register the pre-existing outcome-sealed E170 endpoint after v4 freeze."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "docs/实验结果/Stage2_mature_upstream_20260928"
PUBLIC = Path("/home/yyf/proj/docs/实验结果/E170_primary_cd4_multipanel_precision_20260718")
F2_ROOT = Path("/home/yyf/data/safeconf_external/primary_cd4_perturbseq_2025/isolated/E170")
TRUTH_ROOT = Path("/home/yyf/data/safeconf_v4_e170_confirmation_truth")
PANELS = ("P01", "P02", "P03", "P04")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if TRUTH_ROOT.exists():
        raise RuntimeError("refusing to register after E170 v4 truth has been opened")
    pretruth = json.loads((PUBLIC / "PRETRUTH_RUN_STATUS.json").read_text())
    if pretruth.get("test_targeting_x_values_read") != 0 or pretruth.get("column_unseen_targeting_x_values_read") != 0:
        raise RuntimeError("old E170 test targeting truth is not sealed")
    freeze = json.loads((STAGE / "FINAL_CANDIDATE_FREEZE.json").read_text())
    config = json.loads((STAGE / "FINAL_METHOD_CONFIG.json").read_text())
    if freeze.get("candidate") != "V2_nested_evidence_shrinkage" or config.get("status") != "FINAL_CANDIDATE_FROZEN_PRE_CONFIRMATION":
        raise RuntimeError("SafeConf-v4 final candidate is not frozen")
    competence = pd.read_csv(STAGE / "UPSTREAM_COMPETENCE_V4.csv")
    row = competence.loc[
        competence.asset.eq("E170_primary_CD4_four_panel")
        & competence.upstream_model.eq("scGPT_GEARS_6member_ensemble")
    ]
    if len(row) != 1 or not bool(row.iloc[0].passes_2pct_competence_gate):
        raise RuntimeError("E170 upstream competence gate is absent or failed")

    frozen_files = [
        STAGE / "E170_SAFECONF_V4_CONFIRMATION_CONTRACT.md",
        STAGE / "FINAL_METHOD_CONFIG.json",
        STAGE / "FINAL_CANDIDATE_FREEZE.json",
        ROOT / "tools/scripts/run_safeconf_v4_development.py",
        ROOT / "tools/scripts/build_e168_primary_cd4_isolated_assets.py",
        ROOT / "tools/scripts/build_e170_primary_cd4_panel_assets.py",
        ROOT / "tools/scripts/build_e170_v4_confirmation_truth.py",
        ROOT / "tools/scripts/run_e170_safeconf_v4_confirmation.py",
    ]
    source_lock = json.loads((PUBLIC / "SOURCE_LOCK.json").read_text())
    panels = {}
    audit_rows = []
    for panel in PANELS:
        release = PUBLIC / "pretruth_release" / panel
        f2 = F2_ROOT / panel / "F2_pretruth"
        snapshot = json.loads((release / "PRETRUTH_GATE_SNAPSHOT.json").read_text())
        if snapshot.get("test_targeting_x_values_read") != 0 or snapshot.get("forbidden_column_unseen_x_values_read") != 0:
            raise RuntimeError(f"{panel} pretruth snapshot is not outcome sealed")
        panels[panel] = {
            "prediction_sha256": sha256(release / "arrays/PRETRUTH_PREDICTIONS.npz"),
            "f2_manifest_sha256": sha256(f2 / "MANIFEST.sha256"),
            "old_legacy_gate_status": snapshot.get("status"),
            "n_test_tasks": 600,
            "n_target_clusters": 200,
        }
        for field, raw, eligible, missing, derivation in [
            ("Universal_P", 1.0, 1.0, 0, "frozen ensemble prediction"),
            ("Support", 0.8, 0.8, 120, "two train-donor matched-condition effects"),
            ("Relevance", 0.8, 0.8, 120, "prediction/source cosine and negative RMSE gap"),
            ("Quality", 0.0, 0.0, 600, "not in frozen candidate"),
            ("Conflict", 0.8, 0.8, 120, "two train-donor effect dispersion"),
            ("Content", 0.8, 0.8, 120, "train-donor mean effect magnitude"),
        ]:
            audit_rows.append({
                "dataset_id": "E170_primary_CD4_test", "panel_id": panel,
                "field_name": field, "raw_coverage": raw, "eligible_coverage": eligible,
                "missing_after_eligibility": missing, "n_independent_studies": 1 if eligible else 0,
                "n_independent_contexts": 6 if eligible else 0,
                "derivation": derivation, "leakage_safe": bool(field != "Quality"),
                "audited_after_final_candidate_freeze": True,
                "test_truth_opened_during_audit": False,
            })
    auth = {
        "status": "AUTHORIZED_AFTER_FINAL_CANDIDATE_FREEZE",
        "final_candidate": "V2_nested_evidence_shrinkage",
        "final_candidate_frozen_at_utc": config["final_candidate_frozen_at_utc"],
        "all_four_panels_required": True,
        "test_truth_opened_at_authorization": False,
        "n_test_tasks": 2400,
        "n_target_clusters": 800,
        "n_evaluation_strata": 12,
        "confirmation_level": "new perturbations and held-out donor within the same study",
        "source_path": source_lock["source_path"],
        "source_bytes": source_lock["source_bytes"],
        "source_sha256": source_lock["source_full_sha256"],
        "upstream_competence": row.iloc[0].to_dict(),
        "panels": panels,
        "frozen_file_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in frozen_files
        },
        "rules": {
            "panel_selection": "all P01-P04; no selection by old legacy gate",
            "risk_fit_labels": "validation-donor errors only",
            "test_truth_use": "evaluation only after every risk score is generated",
            "primary_comparison": "Frozen V2 versus Magnitude",
            "v1_role": "pre-specified secondary benchmark",
            "post_result_method_switching": False,
        },
    }
    (STAGE / "E170_V4_CONFIRMATION_AUTHORIZATION.json").write_text(json.dumps(auth, indent=2) + "\n")
    pd.DataFrame(audit_rows).to_csv(STAGE / "E170_PREOPEN_FIELD_AUDIT.csv", index=False)
    print(json.dumps({
        "status": auth["status"], "test_truth_opened": False,
        "n_test_tasks": auth["n_test_tasks"], "n_evaluation_strata": auth["n_evaluation_strata"],
    }, indent=2))


if __name__ == "__main__":
    main()
