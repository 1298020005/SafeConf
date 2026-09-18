from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools/scripts/run_e213_selective_prediction_endpoints.py"
SPEC = importlib.util.spec_from_file_location("e213", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class E213EndpointTests(unittest.TestCase):
    def test_perfect_risk_score_has_perfect_detection_and_review(self) -> None:
        outcome = np.arange(1.0, 21.0)
        label = MODULE.binary_top_label(outcome)
        self.assertTrue(math.isclose(MODULE.binary_auroc(outcome, label), 1.0))
        self.assertTrue(math.isclose(MODULE.average_precision(outcome, label), 1.0))
        self.assertGreater(MODULE.review_utility(outcome, outcome, 0.20), 0.999)

    def test_selective_risk_rewards_removing_large_errors_first(self) -> None:
        outcome = np.arange(1.0, 21.0)
        good = MODULE.selective_risk_auc(outcome, outcome)
        reversed_score = MODULE.selective_risk_auc(-outcome, outcome)
        self.assertLess(good, reversed_score)

    def test_one_sided_formula_never_decreases_numeric_score(self) -> None:
        magnitude = np.array([0.2, 0.7, 0.4])
        safeconf = np.array([0.8, 0.1, 0.6])
        score = magnitude + 0.25 * np.maximum(safeconf - magnitude, 0.0)
        self.assertTrue(np.all(score >= magnitude))


if __name__ == "__main__":
    unittest.main()

