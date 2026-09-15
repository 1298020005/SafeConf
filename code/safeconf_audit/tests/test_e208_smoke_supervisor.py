"""Tests for the E205-to-E208 smoke supervisor gates."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools/scripts/run_e208_after_e205_smoke_supervisor.py"
SPEC = importlib.util.spec_from_file_location("e208_smoke_supervisor", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class E208SmokeSupervisorTests(unittest.TestCase):
    def test_e205_gate_wait_go_and_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            self.assertEqual(MODULE.e205_gate(path)[0], "WAIT")
            path.write_text(
                json.dumps(
                    {
                        "status": "RUNNING",
                        "completed": 7,
                        "permanent_failures": [],
                        "target_truth_access": "NOT_AUTHORIZED",
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(MODULE.e205_gate(path)[0], "WAIT")
            path.write_text(
                json.dumps(
                    {
                        "status": "COMPLETE",
                        "completed": 16,
                        "permanent_failures": [],
                        "target_truth_access": "NOT_AUTHORIZED",
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(MODULE.e205_gate(path)[0], "GO")
            path.write_text(
                json.dumps(
                    {
                        "status": "COMPLETE",
                        "completed": 16,
                        "permanent_failures": ["jurkat/seed_4"],
                        "target_truth_access": "NOT_AUTHORIZED",
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(MODULE.e205_gate(path)[0], "FAIL")

    def test_only_cuda_oom_is_classified_as_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "fit.log").write_text(
                "torch.cuda.OutOfMemoryError: CUDA out of memory", encoding="utf-8"
            )
            self.assertTrue(MODULE.is_oom_failure(root))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "fit.log").write_text(
                "ValueError: missing control mapping", encoding="utf-8"
            )
            self.assertFalse(MODULE.is_oom_failure(root))


if __name__ == "__main__":
    unittest.main()
