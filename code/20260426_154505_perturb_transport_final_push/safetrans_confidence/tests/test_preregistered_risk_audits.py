from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from safetrans_confidence.cli.run_preregistered_risk_audits import (
    FEATURE_GROUPS,
    _NaturalSplineCalibrator,
    _bh_adjust,
    _cross_fitted_expected_error,
    _empirical_upper_tail_p,
    _feature_group_contract,
    _jointly_shuffle_features,
    _paired_bootstrap_e2,
    _rank_target_values,
    _shuffle_task_paired_targets,
)
from safetrans_confidence.cli.run_lopo_third_predictor import ALL_FEATURES


def _paired_source() -> pd.DataFrame:
    rows = []
    for task_index, task in enumerate(("t1", "t2", "t3", "t4")):
        for predictor_index, predictor in enumerate(
            ("V0StrongBaseline", "ContextSimBaseline")
        ):
            rows.append(
                {
                    "dataset_name": "D",
                    "fold_id": 0,
                    "task_key": task,
                    "predictor_name": predictor,
                    "true_error_rmse": 10 * task_index + predictor_index,
                    "f1": 100 * task_index + predictor_index,
                    "f2": -(100 * task_index + predictor_index),
                }
            )
    return pd.DataFrame(rows)


def test_feature_groups_partition_all_fourteen_features() -> None:
    _feature_group_contract()
    grouped = [feature for group in FEATURE_GROUPS.values() for feature in group]
    assert len(grouped) == 14
    assert set(grouped) == set(ALL_FEATURES)


def test_task_target_shuffle_preserves_predictor_error_pairs() -> None:
    source = _paired_source()
    shuffled = _shuffle_task_paired_targets(
        source,
        np.random.default_rng(19),
    )
    shuffled_frame = source[["task_key", "predictor_name"]].copy()
    shuffled_frame["error"] = shuffled
    observed_pairs = {
        tuple(group.sort_values("predictor_name")["true_error_rmse"])
        for _, group in source.groupby("task_key")
    }
    shuffled_pairs = {
        tuple(group.sort_values("predictor_name")["error"])
        for _, group in shuffled_frame.groupby("task_key")
    }
    assert shuffled_pairs == observed_pairs


def test_shuffled_targets_are_ranked_within_fold_and_predictor() -> None:
    source = _paired_source()
    shuffled = _shuffle_task_paired_targets(
        source,
        np.random.default_rng(23),
    )
    ranked = _rank_target_values(source, shuffled)
    for _, indices in source.groupby("predictor_name").groups.items():
        assert sorted(ranked.loc[list(indices)].tolist()) == pytest.approx(
            [0.25, 0.5, 0.75, 1.0]
        )


def test_joint_feature_shuffle_preserves_complete_row_vectors() -> None:
    source = _paired_source()
    shuffled = _jointly_shuffle_features(
        source,
        ["f1", "f2"],
        np.random.default_rng(29),
    )
    for predictor in source["predictor_name"].unique():
        original = source[source["predictor_name"].eq(predictor)]
        permuted = shuffled[shuffled["predictor_name"].eq(predictor)]
        assert sorted(map(tuple, original[["f1", "f2"]].to_numpy())) == sorted(
            map(tuple, permuted[["f1", "f2"]].to_numpy())
        )


def test_natural_spline_clips_out_of_range_predictions() -> None:
    model = _NaturalSplineCalibrator().fit(
        np.linspace(-1.0, 1.0, 9),
        np.linspace(0.0, 2.0, 9) ** 2,
    )
    predictions = model.predict(np.asarray([-10.0, 0.0, 10.0]))
    boundary_predictions = model.predict(np.asarray([-1.0, 0.0, 1.0]))
    assert np.isfinite(predictions).all()
    np.testing.assert_allclose(predictions, boundary_predictions)


def test_cross_fitted_calibration_has_zero_task_overlap() -> None:
    source = pd.DataFrame(
        {
            "task_key": np.repeat([f"t{i}" for i in range(6)], 2),
            "true_error_rmse": np.linspace(0.2, 1.4, 12),
        }
    )
    expected, audit = _cross_fitted_expected_error(
        source,
        np.linspace(-2.0, 2.0, len(source)),
        "isotonic",
        n_splits=3,
    )
    assert np.isfinite(expected).all()
    assert audit["task_overlap"].eq(0).all()


def test_e2_aurc_improvement_direction_is_magnitude_minus_combined() -> None:
    errors = np.arange(1.0, 11.0)
    expected = errors[::-1]
    group = pd.DataFrame(
        {
            "task_key": [f"t{i}" for i in range(10)],
            "true_error_rmse": errors,
            "true_effect_l2_norm": [0.2, 1.7, 0.5, 1.2, 0.9, 2.0, 0.4, 1.4, 0.7, 1.0],
            "expected_error_magnitude": expected,
            "combined_predicted_error": errors,
            "true_residual_error": errors - expected,
            "predicted_residual_error": errors - expected,
        }
    )
    result = _paired_bootstrap_e2(group, n_bootstrap=30, seed=31)
    assert result["aurc_improvement_magnitude_minus_combined_point"] > 0


def test_bh_adjustment_matches_known_example() -> None:
    adjusted = _bh_adjust(pd.Series([0.01, 0.04, 0.03, 0.20]))
    np.testing.assert_allclose(adjusted, [0.04, 0.0533333333, 0.0533333333, 0.20])


def test_empirical_p_does_not_call_undefined_constant_score_significant() -> None:
    assert np.isnan(_empirical_upper_tail_p(np.nan, np.asarray([0.1, 0.2])))
    assert np.isnan(_empirical_upper_tail_p(0.2, np.asarray([np.nan, np.nan])))
    assert _empirical_upper_tail_p(0.5, np.asarray([0.1, 0.2, 0.3])) == 0.25
