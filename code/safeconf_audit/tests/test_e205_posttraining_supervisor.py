from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools/scripts/run_e205_posttraining_supervisor.py"
SPEC = importlib.util.spec_from_file_location("e205_posttraining", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class E205PostTrainingSupervisorTests(unittest.TestCase):
    def test_training_gate_wait_go_and_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "queue.json"
            self.assertEqual(MODULE.training_gate(path)[0], "WAIT")
            path.write_text(
                json.dumps(
                    {
                        "status": "RUNNING",
                        "completed": 7,
                        "target_truth_access": "NOT_AUTHORIZED",
                        "permanent_failures": [],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(MODULE.training_gate(path)[0], "WAIT")
            path.write_text(
                json.dumps(
                    {
                        "status": "COMPLETE",
                        "completed": 16,
                        "target_truth_access": "NOT_AUTHORIZED",
                        "permanent_failures": [],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(MODULE.training_gate(path)[0], "GO")
            value = json.loads(path.read_text(encoding="utf-8"))
            value["permanent_failures"] = [{"target": "K562", "seed": 4}]
            path.write_text(json.dumps(value), encoding="utf-8")
            self.assertEqual(MODULE.training_gate(path)[0], "FAIL")

    def test_prediction_complete_checks_truth_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "K562/seed_1"
            directory.mkdir(parents=True)
            array = directory / "predictions.npy"
            array.write_bytes(b"sealed predictions")
            manifest = {
                "status": "COMPLETE",
                "target": "K562",
                "seed": 1,
                "target_truth_materialized": False,
                "target_expression_nonzero_values_seen": 0,
                "prediction_file": {
                    "bytes": array.stat().st_size,
                    "sha256": MODULE.sha256_file(array),
                },
            }
            (directory / "E205_PREDICTION_RUN.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            self.assertTrue(MODULE.prediction_complete(root, "K562", 1))
            manifest["target_truth_materialized"] = True
            (directory / "E205_PREDICTION_RUN.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            self.assertFalse(MODULE.prediction_complete(root, "K562", 1))


if __name__ == "__main__":
    unittest.main()
