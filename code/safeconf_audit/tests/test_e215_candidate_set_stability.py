from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools/scripts/run_e215_candidate_set_stability.py"
SPEC = importlib.util.spec_from_file_location("e215", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class E215StabilityTests(unittest.TestCase):
    def test_formulas_are_finite_and_one_sided_never_lowers_magnitude(self) -> None:
        magnitude = np.array([0.1, 0.4, 0.3, 0.2])
        safeconf = np.array([0.4, 0.1, 0.2, 0.3])
        scores = MODULE.formulas(magnitude, safeconf)
        magnitude_rank = MODULE.ranks(magnitude)
        self.assertTrue(np.isfinite(scores["fixed_80_20"]).all())
        self.assertTrue(np.all(scores["one_sided_025"] >= magnitude_rank))

    def test_jaccard_and_perfect_spearman(self) -> None:
        self.assertTrue(math.isclose(MODULE.jaccard({1, 2}, {2, 3}), 1 / 3))
        values = np.arange(10.0)
        self.assertTrue(math.isclose(MODULE.spearman(values, values), 1.0))


if __name__ == "__main__":
    unittest.main()

