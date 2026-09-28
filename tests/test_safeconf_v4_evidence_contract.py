"""Consistency checks for the frozen SafeConf-v4 evidence package."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "docs/实验结果/Stage2_mature_upstream_20260928"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_implementation_and_candidate_are_unchanged() -> None:
    config = json.loads((STAGE / "FINAL_METHOD_CONFIG.json").read_text())
    freeze = json.loads((STAGE / "FINAL_CANDIDATE_FREEZE.json").read_text())
    implementation = ROOT / "tools/scripts/run_safeconf_v4_development.py"
    assert config["status"] == "FINAL_CANDIDATE_FROZEN_PRE_CONFIRMATION"
    assert config["final_candidate"] == "V2_nested_evidence_shrinkage"
    assert freeze["candidate"] == config["final_candidate"]
    assert freeze["passed_v2_gate"] is True
    assert digest(implementation) == freeze["implementation_sha256"]


def test_seen_and_sealed_endpoints_are_not_conflated() -> None:
    registry = pd.read_csv(STAGE / "DATA_ROLE_REGISTRY.csv")
    sealed = registry[registry.role.eq("SEALED_CONFIRMATION")]
    assert set(sealed.dataset) == {
        "E208_Jiang24_test", "E247_Kaden_CRISPRa_test", "E258_Feng2025_test",
        "E170_primary_CD4_test",
    }
    assert sealed.truth_version.isna().all()
    assert (~sealed.result_seen.astype(bool)).all()
    assert registry.loc[registry.dataset.eq("E216_Jiang24_resource"), "role"].item() == "SEEN"
    for prefix in ("E208_Jiang24_validation", "E247_Kaden_CRISPRa_validation", "E258_Feng2025_development"):
        assert registry.loc[registry.dataset.eq(prefix), "role"].item() == "SEEN"


def test_confirmation_audit_never_promotes_incompetent_upstream() -> None:
    audit = pd.read_csv(STAGE / "CONFIRMATION_FIELD_AUDIT.csv")
    assert len(audit) == 4
    assert audit.upstream_competence.eq("BLOCKED").all()
    assert (~audit.eligible_for_v4_confirmation.astype(bool)).all()
    unopened = audit[audit.data_role_after_freeze.eq("SEALED_CONFIRMATION")]
    assert (~unopened.test_truth_opened.astype(bool)).all()


def test_development_gate_matches_released_tables() -> None:
    deltas = []
    nonnegative = []
    for upstream in ("txpert_gat", "txpert_exphormer"):
        path = STAGE / "safeconf_v4_development" / upstream
        summary = pd.read_csv(path / "SUMMARY.csv").set_index("method")
        strata = pd.read_csv(path / "STRATUM_RESULTS.csv").pivot(
            index="target", columns="method", values="utility20"
        )
        deltas.append(summary.loc["V2_nested", "utility20"] - summary.loc["Ridge_USR", "utility20"])
        nonnegative.extend((strata["V2_nested"] - strata["Ridge_USR"] >= 0).tolist())
    freeze = json.loads((STAGE / "FINAL_CANDIDATE_FREEZE.json").read_text())
    assert abs(sum(deltas) / len(deltas) - freeze["aggregate"]["delta_u20_macro"]) < 1e-12
    assert abs(sum(nonnegative) / len(nonnegative) - freeze["aggregate"]["nonnegative_strata_fraction"]) < 1e-12


def test_pertema_comparison_uses_registered_contract() -> None:
    status = json.loads((STAGE / "pertema_fair_comparison/RUN_STATUS.json").read_text())
    boot = pd.read_csv(STAGE / "pertema_fair_comparison/PAIRED_BOOTSTRAP.csv")
    assert status["same_tasks"] is True
    assert status["same_outer_label_budget"] is True
    assert status["outer_gene_disjoint"] is True
    rows = boot[boot.comparison.eq("SafeConf_V2_vs_PertEMA")]
    assert len(rows) == 2
    assert (rows.utility_ci95_lower > 0).all()


def test_quality_is_not_claimed_without_eligible_fields() -> None:
    fields = pd.read_csv(STAGE / "FIELD_AVAILABILITY_MATRIX.csv")
    txpert = fields[fields.dataset_id.astype(str).str.contains("TxPert")]
    quality = txpert[txpert.field_name.astype(str).str.contains("quality", case=False, na=False)]
    assert not quality.empty
    assert (quality.eligible_coverage == 0).all()


def test_e190_is_qualified_but_never_promoted_to_confirmation() -> None:
    registry = pd.read_csv(STAGE / "DATA_ROLE_REGISTRY.csv")
    row = registry.loc[registry.dataset.eq("E190_Adamson_to_Replogle_K562")].iloc[0]
    assert row.role == "SEEN"
    assert bool(row.result_seen) is True
    competence = pd.read_csv(STAGE / "UPSTREAM_COMPETENCE_V4.csv")
    gate = competence.loc[
        competence.asset.eq("E190_Adamson_to_Replogle_K562")
        & competence.upstream_model.eq("GEARS_3seed_centroid")
    ].iloc[0]
    assert bool(gate.passes_2pct_competence_gate) is True
    assert gate.relative_macro_error_gap <= 0.02
    status = json.loads((STAGE / "safeconf_v4_development/e190_gears_crossfamily/RUN_STATUS.json").read_text())
    assert status["data_role"] == "SEEN"
    assert status["independent_confirmation"] is False
    assert status["sealed_confirmation_opened"] is False
    assert status["final_candidate_changed"] is False


def test_e170_authorization_is_post_freeze_and_outcome_sealed() -> None:
    auth = json.loads((STAGE / "E170_V4_CONFIRMATION_AUTHORIZATION.json").read_text())
    freeze = json.loads((STAGE / "FINAL_METHOD_CONFIG.json").read_text())
    assert auth["status"] == "AUTHORIZED_AFTER_FINAL_CANDIDATE_FREEZE"
    assert auth["final_candidate"] == freeze["final_candidate"]
    assert auth["all_four_panels_required"] is True
    assert auth["test_truth_opened_at_authorization"] is False
    assert auth["n_test_tasks"] == 2400
    assert auth["upstream_competence"]["passes_2pct_competence_gate"] is True
    audit = pd.read_csv(STAGE / "E170_PREOPEN_FIELD_AUDIT.csv")
    assert set(audit.panel_id) == {"P01", "P02", "P03", "P04"}
    assert (~audit.test_truth_opened_during_audit.astype(bool)).all()
    assert audit.loc[audit.field_name.eq("Quality"), "eligible_coverage"].eq(0).all()


if __name__ == "__main__":
    checks = [
        test_frozen_implementation_and_candidate_are_unchanged,
        test_seen_and_sealed_endpoints_are_not_conflated,
        test_confirmation_audit_never_promotes_incompetent_upstream,
        test_development_gate_matches_released_tables,
        test_pertema_comparison_uses_registered_contract,
        test_quality_is_not_claimed_without_eligible_fields,
        test_e190_is_qualified_but_never_promoted_to_confirmation,
        test_e170_authorization_is_post_freeze_and_outcome_sealed,
    ]
    for check in checks:
        check()
        print(f"PASS {check.__name__}")
