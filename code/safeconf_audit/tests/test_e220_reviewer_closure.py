"""Check the reviewer controls against independent small calculations."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("e220", ROOT / "tools/scripts/run_e220_reviewer_closure.py")
e220 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e220)


def test_ranked_magnitude_preserves_every_within_context_metric():
    frame = pd.DataFrame({"target": ["a"] * 10 + ["b"] * 10})
    x = np.r_[np.arange(1, 11), 100 * np.arange(1, 11)].astype(float)
    y = np.r_[np.arange(1, 11)[::-1], np.arange(1, 11)].astype(float)
    ranks = e220.within_rank(frame, x)
    for idx in frame.groupby("target").indices.values():
        result = e220.metric_matrix(np.array([x[idx], ranks[idx]]), y[idx], np.arange(10))
        np.testing.assert_allclose(result[0], result[1], atol=1e-14)


def test_perfect_and_reverse_order_have_expected_review_metrics():
    y = np.arange(1, 11, dtype=float)
    result = e220.metric_matrix(np.array([y, -y]), y, np.arange(10))
    np.testing.assert_allclose(result[:, 0], [1, -1], atol=1e-14)
    np.testing.assert_allclose(result[:, 1], [1, -1], atol=1e-14)
    np.testing.assert_allclose(result[:, 2], [19 / 55, 3 / 55])
    np.testing.assert_allclose(result[:, 3], [4.5, 6.5])
    assert result[0, 4] < result[1, 4]


def test_spearman_ties_matches_scipy_and_utility_hand_calculation():
    x = np.array([2, 2, 5, 3, 4, 6, 1, 7, 8, 9.])
    y = np.array([1, 3, 2, 7, 9, 4, 5, 2, 6, 8.])
    result = e220.metric_matrix(x[None], y, np.arange(10))[0]
    assert abs(result[0] - spearmanr(x, y).statistic) < 1e-14
    expected = ((6 + 8) / 2 - y.mean()) / ((8 + 9) / 2 - y.mean())
    assert abs(result[1] - expected) < 1e-14


def test_cluster_resampling_keeps_all_contexts_together():
    labels = np.array(["g1", "g1", "g2", "g2", "g3", "g3"])
    selected = e220.cluster_indices(labels, np.random.default_rng(21))
    for left in (0, 2, 4):
        assert np.sum(selected == left) == np.sum(selected == left + 1)


def test_current_released_score_and_batch_sizes():
    frame = pd.read_csv(e220.DEFAULT_INPUT)
    matrices = e220.score_matrices(frame)
    names, values = matrices["exphormer"]
    assert np.max(np.abs(values[names.index("safeconf_m_4to1")] - frame.safeconf_m_4to1)) == 0
    assert len(set(frame.groupby("target").size())) > 1
    residual = frame.family_rms_error ** 2 - frame.family_centroid_rmse ** 2 - frame.family_disagreement ** 2
    assert np.max(np.abs(residual)) < 1e-8
