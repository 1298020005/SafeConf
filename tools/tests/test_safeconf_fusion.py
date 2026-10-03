import numpy as np
import pytest

from tools.safeconf_continual.fusion import (
    OOFReadyRidgeFusion,
    TrainOnlyQuantileScale,
    build_fusion_channels,
    rule_score,
)


def test_quantile_scale_uses_training_values_only():
    scale = TrainOnlyQuantileScale.fit(np.array([1.0, 2.0, 3.0]), "public")
    before = scale.audit()["fit_hash"]
    transformed = scale.transform(np.array([0.0, 2.0, 100.0, np.nan]))
    assert np.allclose(transformed[:3], [0.0, 2 / 3, 1.0])
    assert np.isnan(transformed[3])
    assert scale.audit()["fit_hash"] == before


def test_rule_score_does_not_require_error_labels_and_uses_amplitude_when_missing():
    score, has_public = rule_score(np.array([0.2, 0.8]), np.array([np.nan, 0.1]))
    assert np.allclose(score, [0.2, 0.1])
    assert has_public.tolist() == [0, 1]


def test_fusion_requires_registered_source_and_oof_inputs():
    x = np.array([[0.1, 0.2], [0.8, 0.9], [0.4, 0.3]])
    y = np.array([0.1, 0.9, 0.4])
    with pytest.raises(ValueError, match="register"):
        OOFReadyRidgeFusion().fit(x, y, ["a", "b"], base_scores_are_oof=True)
    with pytest.raises(ValueError, match="fold-out-of-fold"):
        OOFReadyRidgeFusion(label_source="SOURCE_ERRORS").fit(
            x, y, ["a", "b"], base_scores_are_oof=False
        )


def test_fusion_records_label_source_and_target_budget():
    x = np.array([[0.1, 0.2], [0.8, 0.9], [0.4, 0.3], [0.7, 0.6]])
    y = np.array([0.1, 0.9, 0.4, 0.7])
    model = OOFReadyRidgeFusion(
        label_source="SOURCE_ERROR_CDF",
        target_error_budget="none",
    ).fit(x, y, ["shared_q", "public_q"], base_scores_are_oof=True)
    audit = model.audit()
    assert audit["label_source"] == "SOURCE_ERROR_CDF"
    assert audit["target_error_budget"] == "none"
    assert audit["base_scores_are_oof"] is True
    assert np.isfinite(model.predict(x)).all()


def test_missing_channels_have_explicit_flags_not_zero_semantics():
    values, names = build_fusion_channels(
        np.array([0.2, 0.8]),
        np.array([np.nan, 0.4]),
        None,
        None,
    )
    assert names == ["amplitude_q", "public_q", "shared_q", "target_q",
                     "has_public", "has_shared", "has_target"]
    assert np.isnan(values[0, 1])
    assert values[:, 5].tolist() == [0, 0]
    assert values[:, 6].tolist() == [0, 0]
