import numpy as np
import pandas as pd

from safetrans_confidence.cli.run_fast_feature_scoring import (
    BLIND_PRIMARY_FEATURE_COLUMNS,
    EVALUATION_DIAGNOSTIC_COLUMNS,
    IDENTITY_COLUMNS,
    audit_disagreement_predictor_sets,
    compute_blind_primary_features_from_records,
    compute_features_from_records,
)


def _toy_records() -> pd.DataFrame:
    tasks = [
        (1, "task_1", "train", "c1", "p1"),
        (2, "task_2", "train", "c2", "p1"),
        (3, "task_3", "val", "c3", "p2"),
        (4, "task_4", "test", "c4", "p1"),
    ]
    rows = []
    for task_id, task_key, split, context, perturbation in tasks:
        ctrl_key = f"ctrl_{task_id}"
        true_key = f"true_{task_id}"
        for predictor_name in ["V0StrongBaseline", "ContextSimBaseline"]:
            record_id = f"{task_key}:{predictor_name}"
            rows.append(
                {
                    "record_id": record_id,
                    "task_id": task_id,
                    "task_key": task_key,
                    "dataset_name": "ToyDataset",
                    "fold_id": 0,
                    "split": split,
                    "context": context,
                    "perturbation": perturbation,
                    "predictor_name": predictor_name,
                    "predicted_effect_key": f"pred_{record_id}",
                    "true_effect_key": true_key,
                    "target_control_key": ctrl_key,
                    "true_error_rmse": 0.1 * task_id,
                    "true_error_cosine": 0.01 * task_id,
                }
            )
    return pd.DataFrame(rows)


def _toy_arrays(records: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    pred_arrays = {}
    true_arrays = {}
    ctrl_arrays = {}
    for task_id in sorted(records["task_id"].unique()):
        true_arrays[f"true_{task_id}"] = np.array([float(task_id), float(task_id + 1)])
        ctrl_arrays[f"ctrl_{task_id}"] = np.array([float(task_id), 1.0])
    for row in records.to_dict("records"):
        offset = 0.0 if row["predictor_name"] == "V0StrongBaseline" else 0.25
        pred_arrays[str(row["predicted_effect_key"])] = np.array(
            [float(row["task_id"]) + offset, float(row["task_id"] + 1)]
        )
    return pred_arrays, true_arrays, ctrl_arrays


def test_blind_primary_features_do_not_require_true_effect_fields():
    records = _toy_records()
    pred_arrays, _true_arrays, ctrl_arrays = _toy_arrays(records)
    blind_records = records.drop(columns=["true_effect_key", "true_error_rmse", "true_error_cosine"])

    features = compute_blind_primary_features_from_records(
        blind_records,
        pred_arrays,
        ctrl_arrays,
        strict_contract=True,
    )

    assert list(features.columns) == IDENTITY_COLUMNS + BLIND_PRIMARY_FEATURE_COLUMNS
    assert "true_effect_key" not in features.columns
    assert "true_error_rmse" not in features.columns
    assert "historical_residual_risk" not in features.columns
    assert len(features) == len(records)
    assert np.isfinite(features.loc[features["split"].eq("test"), "model_disagreement_rmse"]).all()


def test_legacy_combined_features_keep_diagnostics_separate_from_primary_values():
    records = _toy_records()
    pred_arrays, true_arrays, ctrl_arrays = _toy_arrays(records)

    blind = compute_blind_primary_features_from_records(records, pred_arrays, ctrl_arrays, strict_contract=True)
    combined = compute_features_from_records(
        records,
        pred_arrays,
        true_arrays,
        ctrl_arrays,
        strict_contract=False,
    )
    paired = combined[IDENTITY_COLUMNS + BLIND_PRIMARY_FEATURE_COLUMNS].merge(
        blind,
        on=IDENTITY_COLUMNS,
        suffixes=("_combined", "_blind"),
    )

    assert set(EVALUATION_DIAGNOSTIC_COLUMNS).issubset(combined.columns)
    for col in BLIND_PRIMARY_FEATURE_COLUMNS:
        assert np.allclose(paired[f"{col}_combined"], paired[f"{col}_blind"], equal_nan=True)


def test_strict_blind_features_fail_when_expected_disagreement_predictor_missing():
    records = _toy_records()
    pred_arrays, _true_arrays, ctrl_arrays = _toy_arrays(records)
    missing_contextsim = records[records["predictor_name"].ne("ContextSimBaseline")].copy()

    status = audit_disagreement_predictor_sets(missing_contextsim)
    assert status["status"].eq("missing_expected_predictor").all()
    assert status["missing_predictors"].eq("ContextSimBaseline").all()

    try:
        compute_blind_primary_features_from_records(
            missing_contextsim,
            pred_arrays,
            ctrl_arrays,
            strict_contract=True,
        )
    except ValueError as exc:
        assert "Missing expected disagreement predictor sets" in str(exc)
        assert "ContextSimBaseline" in str(exc)
    else:
        raise AssertionError("strict blind feature scoring should fail when a predictor is missing")


def test_legacy_blind_features_warn_with_nan_when_disagreement_predictor_missing():
    records = _toy_records()
    pred_arrays, _true_arrays, ctrl_arrays = _toy_arrays(records)
    missing_contextsim = records[records["predictor_name"].ne("ContextSimBaseline")].copy()

    features = compute_blind_primary_features_from_records(
        missing_contextsim,
        pred_arrays,
        ctrl_arrays,
        strict_contract=False,
    )

    assert features["model_disagreement_rmse"].isna().all()
