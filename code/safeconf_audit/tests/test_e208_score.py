from __future__ import annotations

import math
import unittest

import numpy as np

from safeconf_audit.e208_score import E208ScoreError, score_state_batch, task_features


class E208ScoreTests(unittest.TestCase):
    def test_task_features_match_registered_identity(self) -> None:
        predictions = np.asarray(
            [[1.0, 2.0], [3.0, 2.0], [1.0, 4.0], [3.0, 4.0]]
        )
        control = np.asarray([1.0, 1.0])
        source = np.asarray([[0.0, 1.0], [2.0, 1.0]])
        result = task_features(predictions, control, source, 20)
        self.assertAlmostEqual(result.family_disagreement, 1.0)
        self.assertAlmostEqual(result.model_source_gap, math.sqrt(0.5))
        self.assertAlmostEqual(result.source_delta_dispersion, math.sqrt(0.5))
        self.assertAlmostEqual(result.negative_log_source_cells, -math.log1p(20))
        self.assertEqual(result.support_context_deficit, 27.0)
        self.assertAlmostEqual(result.predicted_magnitude, math.sqrt(2.5))

    def test_state_score_uses_population_zscore_and_average_ranks(self) -> None:
        rows = np.arange(10.0)[:, None]
        components = np.concatenate(
            [rows, rows * 2.0 + 1.0, rows**2, -rows, np.sqrt(rows + 1.0)], axis=1
        )
        magnitude = np.asarray([0.0, 1.0, 1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        scored = score_state_batch(components, magnitude)
        self.assertTrue(np.allclose(scored["z_components"].mean(axis=0), 0.0))
        self.assertTrue(np.allclose(scored["z_components"].std(axis=0), 1.0))
        self.assertEqual(scored["magnitude_percentile_rank"][1], 0.25)
        self.assertEqual(scored["magnitude_percentile_rank"][2], 0.25)
        expected = (
            4.0 * scored["magnitude_percentile_rank"]
            + scored["safeconf_percentile_rank"]
        ) / 5.0
        self.assertTrue(np.allclose(scored["safeconf_m_4to1"], expected))

    def test_invalid_source_or_constant_state_fails_closed(self) -> None:
        with self.assertRaises(E208ScoreError):
            task_features(np.zeros((4, 3)), np.zeros(3), np.zeros((1, 3)), 10)
        with self.assertRaises(E208ScoreError):
            score_state_batch(np.ones((10, 5)), np.arange(10.0))


if __name__ == "__main__":
    unittest.main()
