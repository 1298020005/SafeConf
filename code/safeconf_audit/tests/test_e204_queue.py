"""Regression tests for the E204 shared-GPU queue supervisor."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools" / "scripts" / "run_e204_formal_training_queue.py"
SPEC = importlib.util.spec_from_file_location("e204_queue", SCRIPT)
QUEUE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(QUEUE)


class TestE204Queue(unittest.TestCase):
    def test_start_job_does_not_precreate_frozen_adapter_run_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "formal" / "K562" / "seed_1_risk"
            log_path = root / "queue.log"
            fake_process = mock.Mock(pid=1234)
            with mock.patch.object(QUEUE.subprocess, "Popen", return_value=fake_process):
                process = QUEUE.start_job(
                    Path("/tmp/run_e204_weighted_training.py"),
                    Path(sys.executable),
                    root,
                    root / "weights.csv",
                    run_dir,
                    "K562",
                    1,
                    "risk",
                    "0",
                    log_path,
                )
            self.assertIs(process, fake_process)
            self.assertTrue(run_dir.parent.is_dir())
            self.assertFalse(run_dir.exists())

    def test_validate_complete_checks_weight_contract_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            checkpoint = run_dir / "checkpoints" / "last.ckpt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"checkpoint")
            run_status = {
                "status": "COMPLETE",
                "kind": "formal",
                "target": "K562",
                "seed": 1,
                "current_epoch": 80,
                "target_perturbed_cells_accessed": 0,
                "target_test_dataset_constructed": False,
                "last_model_path": str(checkpoint),
            }
            weight_status = {
                "status": "COMPLETE",
                "target": "K562",
                "seed": 1,
                "kind": "formal",
                "weight_column": "task_weight",
                "weight_manifest_rows_for_target": 1365,
                "unit_weight_fallback_samples": 0,
                "weights_applied_to_training_only": True,
                "target_expression_opened": False,
                "external_txpert_modified": False,
            }
            (run_dir / "E201_RUN_STATUS.json").write_text(json.dumps(run_status))
            (run_dir / "E204_WEIGHTING_STATUS.json").write_text(json.dumps(weight_status))
            QUEUE.validate_complete(run_dir, "K562", 1, "risk")
            weight_status["unit_weight_fallback_samples"] = 1
            (run_dir / "E204_WEIGHTING_STATUS.json").write_text(json.dumps(weight_status))
            with self.assertRaisesRegex(QUEUE.QueueFailure, "zero_weight_fallback"):
                QUEUE.validate_complete(run_dir, "K562", 1, "risk")

    def test_attempt_count_survives_supervisor_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp)
            for attempt in (1, 2):
                (logs / f"K562_seed1_risk_attempt{attempt}.log").touch()
            self.assertEqual(QUEUE.attempts_from_logs(logs, "K562", 1, "risk"), 2)

    def test_failed_directory_is_preserved_before_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "K562" / "seed_1_risk"
            run_dir.mkdir(parents=True)
            (run_dir / "failure.txt").write_text("kept")
            archived = QUEUE.archive_failed_run(run_dir, root, "K562", 1, "risk", 1)
            self.assertIsNotNone(archived)
            self.assertFalse(run_dir.exists())
            self.assertEqual((archived / "failure.txt").read_text(), "kept")

    def test_detached_training_process_can_be_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            launcher = root / "launcher.py"
            run_dir = root / "run"
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "import time; time.sleep(10)",
                    str(launcher),
                    str(run_dir),
                ]
            )
            try:
                for _ in range(20):
                    if process.pid in QUEUE.training_pids_for(run_dir, launcher):
                        break
                    time.sleep(0.05)
                self.assertIn(process.pid, QUEUE.training_pids_for(run_dir, launcher))
            finally:
                process.terminate()
                process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
