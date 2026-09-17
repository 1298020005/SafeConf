from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools/scripts/run_e212_parameter_landscape.py"
SPEC = importlib.util.spec_from_file_location("e212_parameter_landscape", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class E212ParameterLandscapeTests(unittest.TestCase):
    def test_one_sided_rule_never_lowers_task_score(self) -> None:
        magnitude = np.asarray([0.2, 0.8, 0.4])
        safeconf = np.asarray([0.9, 0.1, 0.4])
        score = MODULE.one_sided(magnitude, safeconf, 0.25)
        self.assertTrue(np.all(score >= magnitude))
        self.assertEqual(score[1], magnitude[1])
        self.assertEqual(score[2], magnitude[2])

    def test_review_utility_is_one_for_oracle_ranking(self) -> None:
        outcome = np.asarray([0.1, 0.3, 0.2, 0.9, 0.5])
        self.assertAlmostEqual(MODULE.review_utility(outcome, outcome), 1.0)


if __name__ == "__main__":
    unittest.main()

