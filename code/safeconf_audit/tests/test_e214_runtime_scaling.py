from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools/scripts/run_e214_runtime_scaling.py"
SPEC = importlib.util.spec_from_file_location("e214", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class E214RuntimeTests(unittest.TestCase):
    def test_kernel_is_finite_and_one_sided_never_lowers_magnitude_rank(self) -> None:
        features = MODULE.feature_arrays(500, MODULE.SEED)
        fixed, one_sided = MODULE.routing_kernel(features)
        magnitude_rank = rankdata(features[0], method="average") / len(features[0])
        self.assertEqual(fixed.shape, (500,))
        self.assertTrue(np.isfinite(fixed).all())
        self.assertTrue(np.isfinite(one_sided).all())
        self.assertTrue(np.all(one_sided >= magnitude_rank))

    def test_single_batch_schema_and_positive_throughput(self) -> None:
        result = MODULE.single_batch_benchmark((100,), 2)
        self.assertEqual(len(result), 1)
        self.assertGreater(float(result.iloc[0].tasks_per_second), 0)
        self.assertEqual(int(result.iloc[0].input_bytes), 100 * 5 * 8)


if __name__ == "__main__":
    unittest.main()

