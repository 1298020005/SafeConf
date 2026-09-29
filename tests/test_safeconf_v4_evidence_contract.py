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
    opened = sealed[sealed.dataset.eq("E170_primary_CD4_test")].iloc[0]
    assert bool(opened.result_seen) is True
    assert isinstance(opened.truth_version, str) and len(opened.truth_version) == 64
    still_sealed = sealed[~sealed.dataset.eq("E170_primary_CD4_test")]
    assert still_sealed.truth_version.isna().all()
    assert (~still_sealed.result_seen.astype(bool)).all()
    assert registry.loc[registry.dataset.eq("E216_Jiang24_resource"), "role"].item() == "SEEN"
    for prefix in ("E208_Jiang24_validation", "E247_Kaden_CRISPRa_validation", "E258_Feng2025_development"):
        assert registry.loc[registry.dataset.eq(prefix), "role"].item() == "SEEN"


def test_confirmation_audit_promotes_only_qualified_e170() -> None:
    audit = pd.read_csv(STAGE / "CONFIRMATION_CANDIDATE_AUDIT.csv")
    assert len(audit) == 5
    e170 = audit[audit.candidate.eq("E170_primary_CD4_four_panel")].iloc[0]
    assert e170.upstream_competence == "PASS"
    assert bool(e170.eligible_for_v4_confirmation) is True
    assert bool(e170.test_truth_opened) is True
    blocked = audit[~audit.candidate.eq("E170_primary_CD4_four_panel")]
    assert blocked.upstream_competence.eq("BLOCKED").all()
    assert (~blocked.eligible_for_v4_confirmation.astype(bool)).all()
    unopened = blocked[blocked.data_role_after_freeze.eq("SEALED_CONFIRMATION")]
    assert (~unopened.test_truth_opened.astype(bool)).all()


def test_confirmation_field_audit_preserves_preopen_definitions() -> None:
    preopen = pd.read_csv(STAGE / "E170_PREOPEN_FIELD_AUDIT.csv")
    released = pd.read_csv(STAGE / "CONFIRMATION_FIELD_AUDIT.csv")
    shared = list(preopen.columns)
    pd.testing.assert_frame_equal(preopen[shared], released[shared], check_dtype=False)
    assert released.confirmation_opened_once.astype(bool).all()
    assert (~released.field_definition_changed_after_open.astype(bool)).all()
    assert released.confirmation_result.eq("GATE_A_PASS").all()


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


def test_e170_one_shot_confirmation_passes_registered_gate_a() -> None:
    result = STAGE / "confirmation/e170_primary_cd4_four_panel"
    status = json.loads((result / "RUN_STATUS.json").read_text())
    gate = json.loads((result / "GATE_A_RESULT.json").read_text())
    summary = pd.read_csv(result / "SUMMARY.csv").set_index("method")
    assert status["one_shot_confirmation"] is True
    assert status["all_four_panels_opened_together"] is True
    assert status["n_confirmation_tasks"] == 2400
    assert status["n_target_clusters"] == 800
    assert status["final_candidate"] == "V2_nested_evidence_shrinkage"
    assert status["final_candidate_changed"] is False
    assert gate["gate_a_pass"] is True
    assert gate["delta_u20_macro"] >= 0.005
    assert gate["nonnegative_strata_fraction"] >= 0.60
    assert summary.loc["V2_nested", "utility20"] > summary.loc["Magnitude_raw", "utility20"]


def test_paper_package_records_route_a_without_overclaiming_gate_b() -> None:
    status = json.loads((STAGE / "PAPER_PACKAGE_STATUS.json").read_text())
    assert status["gate_a"].startswith("PASS_E170")
    assert status["gate_b"] == "NOT_EVALUABLE_NO_FROZEN_QUALITY_FIELDS"
    assert status["paper_route"] == "A_SAFECONF_MAIN_METHOD"


if __name__ == "__main__":
    checks = [
        test_frozen_implementation_and_candidate_are_unchanged,
        test_seen_and_sealed_endpoints_are_not_conflated,
        test_confirmation_audit_promotes_only_qualified_e170,
        test_confirmation_field_audit_preserves_preopen_definitions,
        test_development_gate_matches_released_tables,
        test_pertema_comparison_uses_registered_contract,
        test_quality_is_not_claimed_without_eligible_fields,
        test_e190_is_qualified_but_never_promoted_to_confirmation,
        test_e170_authorization_is_post_freeze_and_outcome_sealed,
        test_e170_one_shot_confirmation_passes_registered_gate_a,
        test_paper_package_records_route_a_without_overclaiming_gate_b,
    ]
    for check in checks:
        check()
        print(f"PASS {check.__name__}")
