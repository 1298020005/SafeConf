from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from safetrans_confidence.data.records import (
    add_legacy_contract_defaults,
    assert_no_feature_label_leakage,
    validate_prediction_record_artifacts,
    validate_prediction_record_contract,
)


def _minimal_records() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "record_id": "r1",
                "task_id": "t1",
                "task_key": "d::c::p",
                "dataset_name": "Haber",
                "fold_id": 0,
                "split": "test",
                "context": "ctxA",
                "perturbation": "pertA",
                "predictor_name": "V0StrongBaseline",
                "predicted_effect_key": "pred1",
                "true_effect_key": "true1",
                "true_error_rmse": 0.2,
                "true_error_cosine": 0.1,
            },
            {
                "record_id": "r2",
                "task_id": "t1",
                "task_key": "d::c::p",
                "dataset_name": "Haber",
                "fold_id": 0,
                "split": "test",
                "context": "ctxA",
                "perturbation": "pertA",
                "predictor_name": "ContextSimBaseline",
                "predicted_effect_key": "pred2",
                "true_effect_key": "true1",
                "true_error_rmse": 0.3,
                "true_error_cosine": 0.2,
            },
        ]
    )


def _formal_records() -> pd.DataFrame:
    records = add_legacy_contract_defaults(_minimal_records())
    records["dataset_group"] = "public_crispr_candidate"
    records["gene_panel_id"] = "hvg_2000_v20260608"
    records["gene_order_hash"] = "sha256:abc123"
    records["normalization_id"] = "pseudobulk_mean_diff_v1"
    return records


def _effect_arrays() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    predicted = {
        "pred1": np.array([0.1, 0.2, 0.3]),
        "pred2": np.array([0.3, 0.2, 0.1]),
    }
    true = {"true1": np.array([0.2, 0.2, 0.2])}
    return predicted, true


def test_contract_missing_columns_are_reported_in_strict_mode() -> None:
    issues = validate_prediction_record_contract(_minimal_records(), strict=True)
    assert any(issue.startswith("missing_contract_columns=") for issue in issues)


def test_legacy_defaults_are_allowed_only_for_non_strict_audits() -> None:
    records = add_legacy_contract_defaults(_minimal_records())
    non_strict_issues = validate_prediction_record_contract(records, strict=False)
    strict_issues = validate_prediction_record_contract(records, strict=True)

    assert "legacy_true_effect_key_record_scoped_check_skipped" in non_strict_issues
    assert any(issue.startswith("unknown_contract_provenance=") for issue in strict_issues)


def test_formal_contract_ready_records_are_valid() -> None:
    issues = validate_prediction_record_contract(_formal_records(), strict=True)
    assert issues == []


def test_inconsistent_true_effect_key_across_predictors_is_detected() -> None:
    records = _formal_records()
    records.loc[records["predictor_name"].eq("ContextSimBaseline"), "true_effect_key"] = "other_true"
    issues = validate_prediction_record_contract(records, strict=True)
    assert any("inconsistent_true_effect_key_for_task" in issue for issue in issues)


def test_inconsistent_gene_order_hash_across_predictors_is_detected() -> None:
    records = _formal_records()
    records.loc[records["predictor_name"].eq("ContextSimBaseline"), "gene_order_hash"] = (
        "sha256:other"
    )
    issues = validate_prediction_record_contract(records, strict=True)
    assert any("inconsistent_gene_order_hash_for_task" in issue for issue in issues)


def test_invalid_split_is_detected() -> None:
    records = _formal_records()
    records.loc[0, "split"] = "holdout"
    issues = validate_prediction_record_contract(records, strict=True)
    assert "invalid_split=holdout" in issues


def test_empty_or_unknown_provenance_is_detected_in_strict_mode() -> None:
    records = _formal_records()
    records.loc[0, "gene_order_hash"] = ""
    records.loc[1, "normalization_id"] = "unknown"
    issues = validate_prediction_record_contract(records, strict=True)

    assert any(issue.startswith("empty_required_values=") for issue in issues)
    assert any(issue.startswith("unknown_contract_provenance=") for issue in issues)


def test_duplicate_task_predictor_rows_are_detected() -> None:
    records = pd.concat([_formal_records(), _formal_records().iloc[[0]]], ignore_index=True)
    records.loc[2, "record_id"] = "r3"
    issues = validate_prediction_record_contract(records, strict=True)
    assert "duplicate_task_predictor_rows" in issues


def test_effect_array_contract_accepts_matching_vectors() -> None:
    predicted, true = _effect_arrays()
    issues = validate_prediction_record_contract(
        _formal_records(),
        strict=True,
        predicted_effects=predicted,
        true_effects=true,
    )
    assert issues == []


def test_missing_effect_array_keys_are_detected() -> None:
    predicted, true = _effect_arrays()
    predicted.pop("pred2")
    true.clear()
    issues = validate_prediction_record_contract(
        _formal_records(),
        strict=True,
        predicted_effects=predicted,
        true_effects=true,
    )
    assert "missing_predicted_effect_arrays=pred2" in issues
    assert "missing_true_effect_arrays=true1" in issues


def test_effect_array_shape_mismatch_is_detected() -> None:
    predicted, true = _effect_arrays()
    predicted["pred2"] = np.array([0.1, 0.2])
    issues = validate_prediction_record_contract(
        _formal_records(),
        strict=True,
        predicted_effects=predicted,
        true_effects=true,
    )
    assert any(issue.startswith("effect_array_shape_mismatch=") for issue in issues)


def test_non_vector_effect_arrays_are_detected() -> None:
    predicted, true = _effect_arrays()
    predicted["pred1"] = np.array([[0.1, 0.2, 0.3]])
    issues = validate_prediction_record_contract(
        _formal_records(),
        strict=True,
        predicted_effects=predicted,
        true_effects=true,
    )
    assert any(issue.startswith("invalid_effect_array_shape=") for issue in issues)


def test_prediction_record_artifacts_validate_csv_and_npz(tmp_path) -> None:
    records = _formal_records()
    predicted, true = _effect_arrays()
    tables = tmp_path / "tables"
    arrays = tmp_path / "input"
    tables.mkdir()
    arrays.mkdir()
    records.to_csv(tables / "PREDICTION_RECORDS.csv", index=False)
    np.savez(arrays / "predicted_effects.npz", **predicted)
    np.savez(arrays / "true_effects.npz", **true)

    issues = validate_prediction_record_artifacts(tmp_path, strict=True)
    assert issues == []


def test_prediction_record_artifacts_require_effect_array_files(tmp_path) -> None:
    records = _formal_records()
    tables = tmp_path / "tables"
    tables.mkdir()
    records.to_csv(tables / "PREDICTION_RECORDS.csv", index=False)

    issues = validate_prediction_record_artifacts(tmp_path, strict=True)
    assert "missing_predicted_effect_array_file" in issues
    assert "missing_true_effect_array_file" in issues


def test_feature_label_leakage_is_blocked() -> None:
    with pytest.raises(ValueError):
        assert_no_feature_label_leakage(["context_similarity_max", "true_error_rmse"])
